"""
Route Planner API: give it a list of stops, it returns the best visiting order.

Install:  pip install fastapi uvicorn httpx
Run:      uvicorn route_planner:app --host 0.0.0.0 --port 8000
Docs/UI:  http://localhost:8000/docs   (interactive tester)

Optional security: set env ROUTE_API_KEY=secret and send header X-API-Key: secret
Uses free OpenStreetMap services (OSRM for road times, Nominatim for geocoding).
For heavy/commercial use, host your own OSRM or swap in Google Routes API
inside get_matrix().
"""
import math
import os
from typing import Optional
from urllib.parse import quote

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

OSRM = "https://router.project-osrm.org"
NOMINATIM = "https://nominatim.openstreetmap.org/search"
UA = {"User-Agent": "route-planner-app/1.0"}

app = FastAPI(title="Route Planner API", version="1.0")


def check_key(x_api_key: Optional[str] = Header(None)):
    key = os.getenv("ROUTE_API_KEY")
    if key and x_api_key != key:
        raise HTTPException(401, "Invalid or missing X-API-Key")


class Stop(BaseModel):
    name: Optional[str] = None
    address: Optional[str] = None  # used if lat/lng not given
    lat: Optional[float] = None
    lng: Optional[float] = None


class RouteRequest(BaseModel):
    start: Stop                      # driver's current position / depot
    stops: list[Stop]                # places to visit (any order)
    end: Optional[Stop] = None       # fixed final destination (optional)
    return_to_start: bool = False    # round trip back to start
    optimize_for: str = "duration"   # "duration" or "distance"


async def geocode(client: httpx.AsyncClient, s: Stop) -> Stop:
    if s.lat is not None and s.lng is not None:
        return s
    if not s.address:
        raise HTTPException(422, "Each stop needs lat/lng or an address")
    r = await client.get(NOMINATIM, params={"q": s.address, "format": "json", "limit": 1}, headers=UA)
    data = r.json()
    if not data:
        raise HTTPException(422, f"Could not find address: {s.address}")
    return Stop(name=s.name or s.address, address=s.address,
                lat=float(data[0]["lat"]), lng=float(data[0]["lon"]))


def haversine(a: Stop, b: Stop) -> float:
    p1, p2 = math.radians(a.lat), math.radians(b.lat)
    dl = math.radians(b.lng - a.lng)
    h = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371000 * math.asin(math.sqrt(h))


async def get_matrix(client, nodes: list[Stop]):
    """Returns (durations_sec, distances_m) matrices using real road data."""
    coords = ";".join(f"{n.lng},{n.lat}" for n in nodes)
    try:
        r = await client.get(f"{OSRM}/table/v1/driving/{coords}",
                             params={"annotations": "duration,distance"}, headers=UA)
        j = r.json()
        if j.get("code") == "Ok" and all(None not in row for row in j["durations"]):
            return j["durations"], j["distances"]
    except Exception:
        pass
    # Fallback: straight-line estimate at ~40 km/h
    dist = [[haversine(a, b) for b in nodes] for a in nodes]
    return [[d / 11.1 for d in row] for row in dist], dist


def cost(order, D, loop):
    c = sum(D[a][b] for a, b in zip(order, order[1:]))
    return c + (D[order[-1]][order[0]] if loop else 0)


def solve(D, n_nodes, has_end, loop):
    """Nearest-neighbour start, then 2-opt improvement. Node 0 = start."""
    last = n_nodes - 1 if has_end else None
    todo = set(range(1, n_nodes - (1 if has_end else 0)))
    order = [0]
    while todo:
        nxt = min(todo, key=lambda j: D[order[-1]][j])
        order.append(nxt)
        todo.remove(nxt)
    if has_end:
        order.append(last)
    hi = len(order) - (1 if has_end else 0)
    improved = True
    while improved:
        improved = False
        for i in range(1, hi - 1):
            for j in range(i + 1, hi):
                cand = order[:i] + order[i:j + 1][::-1] + order[j + 1:]
                if cost(cand, D, loop) < cost(order, D, loop) - 1e-9:
                    order, improved = cand, True
    return order


def maps_url(route: list[Stop]) -> str:
    pt = lambda s: f"{s.lat},{s.lng}"
    url = (f"https://www.google.com/maps/dir/?api=1&travelmode=driving"
           f"&origin={pt(route[0])}&destination={pt(route[-1])}")
    if len(route) > 2:
        url += "&waypoints=" + quote("|".join(pt(s) for s in route[1:-1]), safe=",|")
    return url  # Google Maps handles ~9 waypoints per link


@app.post("/optimize", dependencies=[Depends(check_key)])
async def optimize(req: RouteRequest):
    if not req.stops:
        raise HTTPException(422, "Provide at least one stop")
    if len(req.stops) > 90:
        raise HTTPException(422, "Max 90 stops per request")
    async with httpx.AsyncClient(timeout=30) as client:
        start = await geocode(client, req.start)
        stops = [await geocode(client, s) for s in req.stops]
        end = await geocode(client, req.end) if req.end else None
        nodes = [start] + stops + ([end] if end else [])
        T, Dist = await get_matrix(client, nodes)

    M = T if req.optimize_for == "duration" else Dist
    loop = req.return_to_start and end is None
    order = solve(M, len(nodes), end is not None, loop)
    if loop:
        order.append(0)

    legs = [{"from": nodes[a].name or nodes[a].address, "to": nodes[b].name or nodes[b].address,
             "duration_min": round(T[a][b] / 60, 1), "distance_km": round(Dist[a][b] / 1000, 2)}
            for a, b in zip(order, order[1:])]
    route = [nodes[i] for i in order]
    return {
        "ordered_stops": [s.model_dump() for s in route],
        "legs": legs,
        "total_duration_min": round(sum(l["duration_min"] for l in legs), 1),
        "total_distance_km": round(sum(l["distance_km"] for l in legs), 2),
        "google_maps_url": maps_url(route),
    }


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="theme-color" content="#0b57d0"><meta name="apple-mobile-web-app-capable" content="yes">
<title>Route Planner</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--tx:#1b1d21;--mut:#6b7280;--pr:#0b57d0;--on:#fff;--bd:#e3e6ea}
@media(prefers-color-scheme:dark){:root{--bg:#111316;--card:#1b1e23;--tx:#eceef1;--mut:#9aa1ab;--pr:#7aa7ff;--on:#0b1220;--bd:#2c3036}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--tx);font:16px/1.4 system-ui,sans-serif;padding:max(16px,env(safe-area-inset-top)) 16px max(24px,env(safe-area-inset-bottom))}
main{max-width:560px;margin:auto}h1{font-size:22px;margin:4px 0 14px}
.card{background:var(--card);border:1px solid var(--bd);border-radius:14px;padding:14px;margin-bottom:12px}
label{display:block;font-size:13px;color:var(--mut);margin:0 0 6px}
input[type=text],textarea{width:100%;padding:12px;font:inherit;color:var(--tx);background:var(--bg);border:1px solid var(--bd);border-radius:10px;margin-bottom:12px}
textarea{min-height:130px;resize:vertical}
.row{display:flex;gap:8px}.row input{flex:1}
button,a.btn{display:block;width:100%;text-align:center;text-decoration:none;padding:14px;font:600 16px system-ui;border:0;border-radius:12px;background:var(--pr);color:var(--on);cursor:pointer}
button.sec{background:var(--bg);color:var(--tx);border:1px solid var(--bd);width:auto;padding:0 14px;margin-bottom:12px}
.chk{display:flex;gap:8px;align-items:center;margin-bottom:14px}
ol{padding-left:0;list-style:none;margin:0 0 14px}li{display:flex;gap:10px;padding:10px 0;border-bottom:1px solid var(--bd)}
.n{flex:none;width:26px;height:26px;border-radius:50%;background:var(--pr);color:var(--on);text-align:center;font:600 14px/26px system-ui}
.mut{color:var(--mut);font-size:13px}#msg{color:#d93025;margin:8px 0}
</style></head><body><main>
<h1>Route Planner</h1>
<div class="card">
<label for="start">Start</label>
<div class="row"><input id="start" type="text" placeholder="Address or use my location"><button class="sec" id="loc" type="button">Use GPS</button></div>
<label for="stops">Stops (one per line)</label>
<textarea id="stops" placeholder="350 5th Ave, New York&#10;Brooklyn Museum&#10;Times Square"></textarea>
<label for="end">Final destination (optional)</label>
<input id="end" type="text" placeholder="Leave empty to finish at the last stop">
<div class="chk"><input id="loop" type="checkbox"><label for="loop" style="margin:0">Return to start</label></div>
<button id="go" type="button">Plan best route</button><div id="msg"></div>
</div>
<div id="out"></div>
</main>
<script>
const $=id=>document.getElementById(id);let me=null;
function msg(t){$('msg').textContent=t||''}
$('start').oninput=()=>{me=null};
$('loc').onclick=()=>{navigator.geolocation?navigator.geolocation.getCurrentPosition(p=>{me={lat:p.coords.latitude,lng:p.coords.longitude};$('start').value='My location'},()=>msg('Location unavailable. Type a start address.'),{enableHighAccuracy:true}):msg('GPS not supported')};
async function call(body){
  let key='';try{key=localStorage.getItem('rk')||''}catch(e){}
  let r=await fetch('/optimize',{method:'POST',headers:{'Content-Type':'application/json','X-API-Key':key},body:JSON.stringify(body)});
  if(r.status===401){key=prompt('API key')||'';try{localStorage.setItem('rk',key)}catch(e){}
    r=await fetch('/optimize',{method:'POST',headers:{'Content-Type':'application/json','X-API-Key':key},body:JSON.stringify(body)})}
  const j=await r.json();if(!r.ok)throw new Error(j.detail||'Request failed');return j}
$('go').onclick=async()=>{
  msg('');const lines=$('stops').value.split('\n').map(s=>s.trim()).filter(Boolean);
  const st=$('start').value.trim();
  if(!lines.length)return msg('Add at least one stop.');if(!me&&!st)return msg('Set a start.');
  const body={start:me?{name:'My location',lat:me.lat,lng:me.lng}:{name:st,address:st},
    stops:lines.map(a=>({name:a,address:a})),return_to_start:$('loop').checked};
  const e=$('end').value.trim();if(e)body.end={name:e,address:e};
  $('go').disabled=true;$('go').textContent='Planning...';
  try{
    const j=await call(body),out=$('out');out.textContent='';
    const c=document.createElement('div');c.className='card';
    const h=document.createElement('div');h.style.fontWeight='600';
    h.textContent=j.total_duration_min+' min · '+j.total_distance_km+' km';c.appendChild(h);
    const ol=document.createElement('ol');
    j.ordered_stops.forEach((s,i)=>{
      const li=document.createElement('li'),n=document.createElement('span'),d=document.createElement('div');
      n.className='n';n.textContent=i+1;d.textContent=s.name||s.address;
      if(j.legs[i]){const m=document.createElement('div');m.className='mut';
        m.textContent='next: '+j.legs[i].duration_min+' min, '+j.legs[i].distance_km+' km';d.appendChild(m)}
      li.append(n,d);ol.appendChild(li)});
    c.appendChild(ol);
    const a=document.createElement('a');a.className='btn';a.href=j.google_maps_url;a.target='_blank';a.rel='noopener';
    a.textContent='Start navigation';c.appendChild(a);out.appendChild(c);
  }catch(err){msg(err.message)}
  $('go').disabled=false;$('go').textContent='Plan best route';
};
</script></body></html>"""


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home():
    return PAGE


@app.get("/health")
def health():
    return {"status": "ok"}


# Example request body for POST /optimize:
# {
#   "start": {"name": "Depot", "lat": 40.7128, "lng": -74.0060},
#   "stops": [
#     {"name": "Customer A", "address": "350 5th Ave, New York"},
#     {"name": "Customer B", "address": "Brooklyn Museum, New York"},
#     {"name": "Customer C", "lat": 40.7484, "lng": -73.9857}
#   ],
#   "return_to_start": true
# }
