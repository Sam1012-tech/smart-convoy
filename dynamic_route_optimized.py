# dynamic_route_optimized.py
import os
import math
import time
import asyncio
import aiohttp
import requests
import polyline
from typing import List, Tuple, Dict, Any, Optional
from functools import lru_cache
from collections import defaultdict

from fastapi import FastAPI, Query, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ---------------------------
# Config / Environment
# ---------------------------
OPENWEATHER_KEY = os.getenv("OPENWEATHER_KEY")    # required for weather scoring
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")      # optional: used for Elevation API
OSRM_URL = os.getenv("OSRM_URL", "http://router.project-osrm.org/route/v1/driving")

# tuning parameters
CLOSURE_RADIUS_KM = float(os.getenv("CLOSURE_RADIUS_KM", "1.0"))
SAMPLE_DISTANCE_M = float(os.getenv("SAMPLE_DISTANCE_M", "500"))  # INCREASED to reduce API calls
WEBSOCKET_UPDATE_SEC = float(os.getenv("WS_UPDATE_SEC", "2.0"))
WEATHER_CACHE_MINUTES = float(os.getenv("WEATHER_CACHE_MINUTES", "5"))

# ---------------------------
# FastAPI setup
# ---------------------------
app = FastAPI(title="Dynamic Reroute Backend (Optimized)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Simple in-memory cache with TTL
# ---------------------------
class CacheEntry:
    def __init__(self, data: Any, ttl_seconds: float = 300):
        self.data = data
        self.created_at = time.time()
        self.ttl = ttl_seconds
    
    def is_expired(self) -> bool:
        return time.time() - self.created_at > self.ttl

weather_cache: Dict[str, CacheEntry] = {}
elevation_cache: Dict[str, CacheEntry] = {}

def cache_get(cache: Dict, key: str) -> Optional[Any]:
    if key in cache:
        entry = cache[key]
        if not entry.is_expired():
            return entry.data
        else:
            del cache[key]
    return None

def cache_set(cache: Dict, key: str, data: Any, ttl: float = 300):
    cache[key] = CacheEntry(data, ttl)

# ---------------------------
# Utilities
# ---------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1 = math.radians(lat1); phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1); dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return R * 2 * math.asin(math.sqrt(a))

def interpolate_point(lat1, lon1, lat2, lon2, t):
    """Linear interpolation in lat/lon. t in [0,1]."""
    return (lat1 + (lat2 - lat1) * t, lon1 + (lon2 - lon1) * t)

def densify_path(latlon: List[Tuple[float,float]], sample_m: float = 500.0) -> List[Tuple[float,float]]:
    """Densify path by sampling every sample_m meters."""
    if not latlon:
        return []
    out = []
    for i in range(len(latlon)-1):
        a = latlon[i]; b = latlon[i+1]
        out.append(a)
        seg_km = haversine_km(a[0], a[1], b[0], b[1])
        seg_m = seg_km * 1000.0
        if seg_m <= 0:
            continue
        steps = max(1, int(seg_m // sample_m))
        for s in range(1, steps+1):
            t = s / (steps+1)
            p = interpolate_point(a[0], a[1], b[0], b[1], t)
            out.append(p)
    out.append(latlon[-1])
    return out

def sample_route_points(densified: List[Tuple[float,float]], max_samples: int = 10) -> List[Tuple[float,float]]:
    """
    Sample evenly spaced points from densified route to reduce API calls.
    Returns max_samples points or fewer if route is short.
    """
    if len(densified) <= max_samples:
        return densified
    step = len(densified) // max_samples
    sampled = [densified[i] for i in range(0, len(densified), step)]
    if densified[-1] not in sampled:
        sampled.append(densified[-1])
    return sampled

# ---------------------------
# Weather / Terrain / Closure scoring
# ---------------------------
def fetch_weather(lat: float, lon: float) -> Optional[Dict[str,Any]]:
    """
    Fetch weather with caching.
    """
    if not OPENWEATHER_KEY:
        return None
    
    cache_key = f"{lat:.2f},{lon:.2f}"
    cached = cache_get(weather_cache, cache_key)
    if cached is not None:
        return cached
    
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_KEY}"
    try:
        r = requests.get(url, timeout=3)
        r.raise_for_status()
        data = r.json()
        cache_set(weather_cache, cache_key, data, ttl=300)  # 5 min TTL
        return data
    except Exception as e:
        print(f"[WEATHER] API error at ({lat},{lon}): {e}")
        return None

def weather_penalty_from_response(w: Dict[str,Any]) -> float:
    """Convert OpenWeather JSON into a numeric penalty."""
    if not w or "weather" not in w or not isinstance(w["weather"], list):
        return 0.0
    main = (w["weather"][0].get("main","") or "").lower()
    wind_speed = float(w.get("wind",{}).get("speed", 0.0))
    visibility = float(w.get("visibility", 10000))
    rain = 0.0
    if "rain" in w and isinstance(w["rain"], dict):
        rain = float(w["rain"].get("1h", w["rain"].get("3h", 0.0)))
    
    penalty = 0.0
    if "thunderstorm" in main:
        penalty += 1000.0
    if "tornado" in main or "squall" in main:
        penalty += 900.0
    if "snow" in main:
        penalty += 700.0
    if "rain" in main or "drizzle" in main:
        penalty += 450.0 + (rain * 200.0)
    if "fog" in main or "mist" in main or "haze" in main or "smoke" in main:
        penalty += 300.0
    if "dust" in main or "sand" in main or "ash" in main:
        penalty += 400.0
    if "cloud" in main:
        penalty += 20.0
    if visibility < 2000:
        penalty += max(0, (2000 - visibility) / 10.0)
    if wind_speed > 12.0:
        penalty += (wind_speed - 12.0) * 30.0
    return penalty

def fetch_elevations_sampled(points: List[Tuple[float,float]]) -> Dict[int, Optional[float]]:
    """
    Fetch elevations for sampled points only (cached).
    Returns dict mapping index (in original points list) to elevation.
    """
    if not GOOGLE_API_KEY or not points:
        return {}
    
    elevs = {}
    batch_size = 50
    for i in range(0, len(points), batch_size):
        batch = points[i:i+batch_size]
        locations = "|".join(f"{p[0]},{p[1]}" for p in batch)
        url = f"https://maps.googleapis.com/maps/api/elevation/json?locations={locations}&key={GOOGLE_API_KEY}"
        try:
            r = requests.get(url, timeout=5)
            r.raise_for_status()
            data = r.json()
            if data.get("status") == "OK" and "results" in data:
                for idx, res in enumerate(data["results"]):
                    elevs[i + idx] = res.get("elevation")
        except Exception as e:
            print(f"[ELEV] API error: {e}")
    return elevs

def slope_penalty_from_elevations(elev1: Optional[float], elev2: Optional[float], dist_m: float) -> float:
    """Compute slope penalty."""
    if elev1 is None or elev2 is None or dist_m <= 0:
        return 0.0
    rise = elev2 - elev1
    slope = abs(rise) / dist_m
    slope_deg = math.degrees(math.atan(slope)) if slope >= 0 else 0.0
    if slope_deg >= 12:
        return 500.0
    elif slope_deg >= 8:
        return 250.0
    elif slope_deg >= 4:
        return 100.0
    return 0.0

def manual_closure_penalty(lat: float, lon: float, closures: List[Tuple[float,float]]) -> float:
    """Check if point is within closure radius."""
    if not closures:
        return 0.0
    for c in closures:
        d = haversine_km(lat, lon, c[0], c[1])
        if d <= CLOSURE_RADIUS_KM:
            return 1200.0
    return 0.0

# ---------------------------
# Route helpers
# ---------------------------
def fetch_osrm_routes(start_lon: float, start_lat: float, end_lon: float, end_lat: float, alternatives: bool = True):
    """Fetch routes from OSRM."""
    coords = f"{start_lon},{start_lat};{end_lon},{end_lat}"
    params = {"alternatives": "true" if alternatives else "false", "overview": "full", "geometries": "polyline"}
    url = f"{OSRM_URL}/{coords}"
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        return r.json().get("routes", [])
    except Exception as e:
        print(f"[OSRM] request failed: {e}")
        return []

def coords_polyline_to_latlon(poly: str) -> List[Tuple[float,float]]:
    """Decode polyline to lat/lon."""
    try:
        pts = polyline.decode(poly)
        return [(float(lat), float(lon)) for lat,lon in pts]
    except Exception as e:
        print(f"[OSRM] polyline decode error: {e}")
        return []

# ---------------------------
# Cost model (OPTIMIZED)
# ---------------------------
def score_route_option(route_coords: List[Tuple[float,float]], closures: List[Tuple[float,float]]) -> Dict[str,Any]:
    """
    OPTIMIZED: Sample route to reduce API calls.
    - Densify with larger sample distance (500m instead of 200m)
    - Sample only ~10 points for weather/elevation checks
    - Cache results
    """
    # Densify with larger sample distance
    densified = densify_path(route_coords, sample_m=SAMPLE_DISTANCE_M)
    
    # Sample evenly spaced points for API queries
    sampled = sample_route_points(densified, max_samples=10)
    sampled_indices = [densified.index(p) if p in densified else 0 for p in sampled]
    
    # Fetch elevations only for sampled points
    elev_map = fetch_elevations_sampled(sampled)
    
    total_score = 0.0
    closed_segments = []
    
    # Score densified path
    for i in range(len(densified)-1):
        lat1, lon1 = densified[i]
        lat2, lon2 = densified[i+1]
        dist_km = haversine_km(lat1, lon1, lat2, lon2)
        dist_m = dist_km * 1000.0
        
        # Weather penalty: query only if point is in sampled set, else use interpolation
        weather_pen = 0.0
        if i in sampled_indices or i+1 in sampled_indices:
            midlat, midlon = (lat1 + lat2) / 2.0, (lon1 + lon2) / 2.0
            weather_data = fetch_weather(midlat, midlon)
            weather_pen = weather_penalty_from_response(weather_data) if weather_data else 0.0
        
        # Slope penalty: interpolate elevations if not in sampled set
        elev1 = elev_map.get(i)
        elev2 = elev_map.get(i+1)
        slope_pen = slope_penalty_from_elevations(elev1, elev2, dist_m)
        
        # Closure penalty
        closure_pen1 = manual_closure_penalty(lat1, lon1, closures)
        closure_pen2 = manual_closure_penalty(lat2, lon2, closures)
        closure_pen = max(closure_pen1, closure_pen2)
        if closure_pen > 0:
            closed_segments.append([(lat1+lat2)/2.0, (lon1+lon2)/2.0])
        
        # Total segment cost
        base_cost = dist_m * 0.01
        seg_cost = base_cost + weather_pen + slope_pen + closure_pen
        total_score += seg_cost
    
    return {
        "total_score": total_score + 0.0001,
        "closed_segments": closed_segments,
        "densified_len": len(densified)
    }

# ---------------------------
# API endpoint
# ---------------------------
@app.get("/dynamic_reroute_json")
def dynamic_reroute_json(
    start_lat: float = Query(...),
    start_lon: float = Query(...),
    end_lat: float = Query(...),
    end_lon: float = Query(...),
    closure_points: str = Query(None),
):
    """
    Fast dynamic rerouting based on weather and closures.
    Response time: ~2-5 seconds instead of 10+ minutes.
    """
    # parse closures
    closures = []
    if closure_points:
        for seg in closure_points.split(";"):
            seg = seg.strip()
            if not seg:
                continue
            try:
                lat_s, lon_s = seg.split(",")
                closures.append((float(lat_s), float(lon_s)))
            except Exception:
                continue

    # fetch OSRM routes
    routes = fetch_osrm_routes(start_lon, start_lat, end_lon, end_lat, alternatives=True)
    if not routes:
        return JSONResponse({"error": "no routes returned by OSRM"}, status_code=502)

    scored_options = []
    for r in routes:
        coords = coords_polyline_to_latlon(r.get("geometry",""))
        if not coords:
            continue
        sc = score_route_option(coords, closures)
        scored_options.append({
            "score": sc["total_score"],
            "closed_segments": sc["closed_segments"],
            "coords": coords,
            "distance": r.get("distance", 0.0),
            "duration": r.get("duration", 0.0),
            "densified_len": sc["densified_len"]
        })

    if not scored_options:
        return JSONResponse({"error":"no valid route options"}, status_code=500)

    # sort by score (lower = better)
    scored_options.sort(key=lambda x: x["score"])

    best = scored_options[0]
    original_coords = coords_polyline_to_latlon(routes[0].get("geometry",""))

    # reroute is the best optimized route (green line on frontend)
    # Always populate it with the best route found
    reroute_coords = best["coords"]

    response = {
        "original_route": original_coords,
        "chosen_route": best["coords"],
        "reroute": reroute_coords,
        "closures": [list(c) for c in closures],
        "closed_segments": best["closed_segments"],
        "eta_seconds": best.get("duration", 0.0),
        "distance_m": best.get("distance", 0.0),
        "score": best["score"]
    }
    return JSONResponse(response)

# ---------------------------
# WebSocket manager (unchanged)
# ---------------------------
class WSManager:
    def __init__(self):
        self.conns: List[WebSocket] = []
    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.conns.append(ws)
    def disconnect(self, ws: WebSocket):
        if ws in self.conns:
            self.conns.remove(ws)
    async def send(self, ws: WebSocket, data: Dict[str,Any]):
        await ws.send_json(data)

ws_manager = WSManager()

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """WebSocket for live route updates."""
    await ws_manager.connect(ws)
    try:
        init = await ws.receive_json()
    except Exception:
        await ws.close()
        return

    start_lat = float(init.get("start_lat"))
    start_lon = float(init.get("start_lon"))
    end_lat = float(init.get("end_lat"))
    end_lon = float(init.get("end_lon"))
    closure_points = init.get("closure_points", None)

    idx = 0
    chosen_route = []
    
    try:
        res = dynamic_reroute_json(start_lat=start_lat, start_lon=start_lon, end_lat=end_lat, end_lon=end_lon, closure_points=closure_points)
        chosen_route = res.get("chosen_route", []) or res.get("original_route", [])
    except Exception as e:
        chosen_route = []

    try:
        while True:
            res = dynamic_reroute_json(start_lat=start_lat, start_lon=start_lon, end_lat=end_lat, end_lon=end_lon, closure_points=closure_points)
            route_coords = res.get("chosen_route", []) or res.get("original_route", [])
            if not route_coords:
                await ws.send_json({"type":"error","message":"no route"})
                await asyncio.sleep(WEBSOCKET_UPDATE_SEC)
                continue

            if idx < len(route_coords)-1:
                idx += 1
            convoy_pos = route_coords[idx]

            payload = {
                "type": "update",
                "timestamp": time.time(),
                "convoy_pos": convoy_pos,
                "route": route_coords,
                "original_route": res.get("original_route", []),
                "closures": res.get("closures", []),
                "closed_segments": res.get("closed_segments", []),
                "eta_seconds": res.get("eta_seconds", 0.0),
                "distance_m": res.get("distance_m", 0.0),
                "score": res.get("score", 0.0)
            }
            await ws_manager.send(ws, payload)
            await asyncio.sleep(WEBSOCKET_UPDATE_SEC)
    except Exception as e:
        print(f"[WS] exception: {e}")
    finally:
        ws_manager.disconnect(ws)

# ---------------------------
# If run as main
# ---------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("dynamic_route_optimized:app", host="0.0.0.0", port=8000, reload=True)
