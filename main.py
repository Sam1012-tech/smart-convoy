# main.py
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse
import requests, random, joblib, os
from datetime import datetime
import folium
from folium.plugins import MeasureControl, Fullscreen, MiniMap, Draw
import math
import heapq

app = FastAPI(title="CR(S)² Convoy Route Management System - Enhanced")

# -------------------------- Utilities --------------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    # return distance in kilometers
    R = 6371.0
    phi1 = math.radians(lat1); phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1); dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return R * 2 * math.asin(math.sqrt(a))

def ensure_model_loaded():
    """
    Tries to load model and feature list saved as:
      - eta_model.pkl
      - eta_features.pkl
    Returns (model, features) or (None, None) if absent.
    """
    model_path = "eta_model.pkl"
    features_path = "eta_features.pkl"
    if os.path.exists(model_path) and os.path.exists(features_path):
        try:
            model = joblib.load(model_path)
            features = joblib.load(features_path)
            return model, features
        except Exception as e:
            print("Model load failed:", e)
            return None, None
    return None, None

MODEL, MODEL_FEATURES = ensure_model_loaded()
if MODEL:
    print("Loaded ETA model with features:", MODEL_FEATURES)
else:
    print("No ETA model found — /predict_eta will use fallback heuristic")

# -------------------------- Basic routes --------------------------------
@app.get("/")
def home():
    return {"message": "CR(S)² Convoy Route Management System Active!"}

@app.get("/plan_route")
def plan_route(origin: str, destination: str):
    return {
        "origin": origin,
        "destination": destination,
        "distance_km": 250,
        "estimated_time_hr": 5.5,
        "recommended_route": ["Base Camp", "Checkpoint A", "Checkpoint B", "Destination"]
    }

@app.get("/optimize_route")
def optimize_route(terrain: str = "plain", traffic_level: int = 1, priority: str = "normal"):
    delay = traffic_level * random.uniform(0.5, 1.5)
    if priority == "high":
        delay *= 0.7
    suggestion = "Alternate route via NH-44" if delay > 2 else "Proceed with current route"
    return {
        "terrain": terrain,
        "traffic_level": traffic_level,
        "estimated_delay_hr": round(delay, 2),
        "suggestion": suggestion
    }

@app.get("/convoy_status")
def convoy_status(convoy_id: int = 101):
    statuses = ["On Route", "Delayed", "Reached Destination", "Waiting at Checkpoint"]
    random_status = random.choice(statuses)
    progress = random.randint(10, 100)
    return {
        "convoy_id": convoy_id,
        "status": random_status,
        "progress_percent": progress,
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

# -------------------------- A* demo (unchanged) -------------------------
graph = {
    "Base": {"A": 2, "B": 5},
    "A": {"Base": 2, "C": 4, "D": 7},
    "B": {"Base": 5, "D": 3},
    "C": {"A": 4, "D": 1, "Destination": 5},
    "D": {"A": 7, "B": 3, "C": 1, "Destination": 2},
    "Destination": {"C": 5, "D": 2}
}
def heuristic(node, goal):
    distances = {"Base": 10, "A": 8, "B": 6, "C": 3, "D": 1, "Destination": 0}
    return distances.get(node, 0)
def a_star(start, goal):
    queue = [(0, start, [])]
    visited = set()
    while queue:
        (cost, node, path) = heapq.heappop(queue)
        if node in visited: continue
        path = path + [node]
        if node == goal:
            return (cost, path)
        visited.add(node)
        for neighbor, weight in graph.get(node, {}).items():
            if neighbor not in visited:
                total_cost = cost + weight + heuristic(neighbor, goal)
                heapq.heappush(queue, (total_cost, neighbor, path))
    return (float("inf"), [])
@app.get("/find_best_route")
def find_best_route(start: str = "Base", goal: str = "Destination"):
    total_cost, best_path = a_star(start, goal)
    if best_path:
        return {"start": start, "goal": goal, "best_path": best_path, "estimated_distance": total_cost}
    else:
        return {"error": "No route found"}

# -------------------------- Visualize route (Folium + OSRM) --------------
@app.get("/visualize_route", response_class=HTMLResponse)
def visualize_route():
    try:
        # default demo points (Delhi Cantonment -> Gurugram)
        points = [
            [28.6139, 77.2090],
            [28.5865, 77.1660],
            [28.5355, 77.2167],
            [28.4595, 77.0266]
        ]
        coords_str = ";".join([f"{lon},{lat}" for lat, lon in points])
        url = f"https://router.project-osrm.org/route/v1/driving/{coords_str}?overview=full&geometries=geojson"
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if "routes" not in data or not data["routes"]:
            return HTMLResponse("<h3>No route returned from OSRM</h3>")
        route = data["routes"][0]["geometry"]["coordinates"]
        route_latlon = [[c[1], c[0]] for c in route]
        midpoint = route_latlon[len(route_latlon)//2]
        m = folium.Map(location=midpoint, zoom_start=11, tiles="CartoDB dark_matter")
        folium.PolyLine(route_latlon, color="red", weight=6, opacity=0.8).add_to(m)
        folium.Marker(points[0], popup="Start", icon=folium.Icon(color="green")).add_to(m)
        folium.Marker(points[-1], popup="Destination", icon=folium.Icon(color="red")).add_to(m)
        for i, pt in enumerate(points[1:-1], start=1):
            folium.Marker(pt, popup=f"Checkpoint {i}", icon=folium.Icon(color="orange")).add_to(m)
        m.add_child(MeasureControl(primary_length_unit='kilometers'))
        m.add_child(Fullscreen(position='topright'))
        MiniMap(toggle_display=True).add_to(m)
        Draw(export=True).add_to(m)
        return HTMLResponse(content=m._repr_html_())
    except Exception as e:
        return HTMLResponse(f"<h3 style='color:red;'>Error: {e}</h3>")

# -------------------------- Real route endpoint (OSRM-based) ----------
@app.get("/real_route", response_class=HTMLResponse)
def real_route(start_lat: float = Query(...), start_lon: float = Query(...),
               end_lat: float = Query(...), end_lon: float = Query(...),
               traffic_level: int = 1, terrain: str = "plain", vehicle_type: str = "truck", priority: str = "normal"):
    """
    Returns an HTML map (Folium) showing the real road route between two points using OSRM with ETA predictions.
    Example:
      http://127.0.0.1:8000/real_route?start_lat=12.97&start_lon=77.59&end_lat=12.98&end_lon=77.60&traffic_level=1
    """
    try:
        coords_str = f"{start_lon},{start_lat};{end_lon},{end_lat}"
        url = f"https://router.project-osrm.org/route/v1/driving/{coords_str}?overview=full&geometries=geojson"
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        if "routes" not in data or not data["routes"]:
            return HTMLResponse("<h3>No route returned from OSRM</h3>")
        route = data["routes"][0]["geometry"]["coordinates"]
        route_latlon = [[c[1], c[0]] for c in route]
        total_distance_m = data["routes"][0].get("distance", 0)
        total_distance_km = total_distance_m / 1000
        
        # Predict ETA for the entire route (use internal helper to get dict)
        eta_response_data = compute_eta(
            start_lat=start_lat, start_lon=start_lon,
            end_lat=end_lat, end_lon=end_lon,
            traffic_level=traffic_level, terrain=terrain,
            vehicle_type=vehicle_type, priority=priority
        )
        eta_minutes = eta_response_data.get("eta_minutes", 0)
        
        # Calculate arrival time
        from datetime import datetime, timedelta
        departure_time = datetime.now()
        arrival_time = departure_time + timedelta(minutes=eta_minutes)
        
        midpoint = route_latlon[len(route_latlon)//2]
        m = folium.Map(location=midpoint, zoom_start=13, tiles="cartodb positron")
        
        # Add route polyline
        folium.PolyLine(route_latlon, color="red", weight=6, opacity=0.8).add_to(m)
        
        # Add start marker with ETA info
        start_popup = f"""
        <div style="font-family: Arial; width: 250px;">
            <h4 style="margin: 5px 0; color: #2c3e50;">Start Point</h4>
            <p style="margin: 5px 0;"><b>Departure:</b> {departure_time.strftime('%H:%M:%S')}</p>
            <p style="margin: 5px 0;"><b>Distance:</b> {total_distance_km:.2f} km</p>
            <p style="margin: 5px 0;"><b>Est. Duration:</b> {eta_minutes:.0f} min</p>
            <p style="margin: 5px 0;"><b>Est. Arrival:</b> {arrival_time.strftime('%H:%M:%S')}</p>
            <p style="margin: 5px 0;"><b>Terrain:</b> {terrain}</p>
            <p style="margin: 5px 0;"><b>Traffic Level:</b> {traffic_level}</p>
        </div>
        """
        folium.Marker(
            [start_lat, start_lon], 
            popup=folium.Popup(start_popup, max_width=300),
            icon=folium.Icon(color="green", prefix="fa", icon="play")
        ).add_to(m)
        
        # Add end marker
        end_popup = f"""
        <div style="font-family: Arial; width: 250px;">
            <h4 style="margin: 5px 0; color: #2c3e50;">Destination</h4>
            <p style="margin: 5px 0;"><b>Est. Arrival:</b> {arrival_time.strftime('%H:%M:%S')}</p>
        </div>
        """
        folium.Marker(
            [end_lat, end_lon], 
            popup=folium.Popup(end_popup, max_width=300),
            icon=folium.Icon(color="red", prefix="fa", icon="flag-checkered")
        ).add_to(m)
        
        # Generate 3 checkpoints along route if route has enough points
        if len(route_latlon) > 4:
            idxs = [len(route_latlon)//4, len(route_latlon)//2, 3*len(route_latlon)//4]
            for i, idx in enumerate(idxs, 1):
                lat, lon = route_latlon[idx]
                # Calculate ETA to this checkpoint
                checkpoint_eta_response = compute_eta(
                    start_lat=start_lat, start_lon=start_lon,
                    end_lat=lat, end_lon=lon,
                    traffic_level=traffic_level, terrain=terrain,
                    vehicle_type=vehicle_type, priority=priority
                )
                checkpoint_eta_min = checkpoint_eta_response.get("eta_minutes", 0)
                checkpoint_arrival = departure_time + timedelta(minutes=checkpoint_eta_min)
                
                checkpoint_popup = f"""
                <div style="font-family: Arial; width: 250px;">
                    <h4 style="margin: 5px 0; color: #2c3e50;">Checkpoint {i}</h4>
                    <p style="margin: 5px 0;"><b>Est. Time to CP:</b> {checkpoint_eta_min:.0f} min</p>
                    <p style="margin: 5px 0;"><b>Est. Arrival:</b> {checkpoint_arrival.strftime('%H:%M:%S')}</p>
                    <p style="margin: 5px 0;"><b>Location:</b> ({lat:.4f}, {lon:.4f})</p>
                </div>
                """
                folium.Marker(
                    [lat, lon], 
                    popup=folium.Popup(checkpoint_popup, max_width=300),
                    icon=folium.Icon(color="orange", prefix="fa", icon="circle")
                ).add_to(m)
        
        # Add measure control
        m.add_child(MeasureControl(primary_length_unit='kilometers'))
        m.add_child(Fullscreen(position='topright'))
        MiniMap(toggle_display=True).add_to(m)
        Draw(export=True).add_to(m)
        
        return HTMLResponse(content=m._repr_html_())
    except Exception as e:
        return HTMLResponse(f"<h3 style='color:red;'>Error: {e}</h3>")

# -------------------------- ETA prediction endpoint ---------------------
@app.get("/predict_eta")
def predict_eta(start_lat: float = Query(...), start_lon: float = Query(...),
                end_lat: float = Query(...), end_lon: float = Query(...),
                traffic_level: int = 1, terrain: str = "plain", vehicle_type: str = "truck", priority: str = "normal"):
    """
    Predict ETA (minutes) for a route using loaded ML model if available, else fallback heuristic.
    Example:
    /predict_eta?start_lat=28.6139&start_lon=77.2090&end_lat=28.4595&end_lon=77.0266&traffic_level=2
    """
    # For backward compatibility keep route as wrapper; actual computation is in compute_eta
    result = compute_eta(
        start_lat=start_lat, start_lon=start_lon,
        end_lat=end_lat, end_lon=end_lon,
        traffic_level=traffic_level, terrain=terrain,
        vehicle_type=vehicle_type, priority=priority
    )
    return JSONResponse(result)


def compute_eta(start_lat: float, start_lon: float,
                end_lat: float, end_lon: float,
                traffic_level: int = 1, terrain: str = "plain", vehicle_type: str = "truck", priority: str = "normal"):
    """
    Compute ETA and return a plain dict. This helper is used internally by routes
    (e.g. `/real_route`) so they can get a Python dict instead of a Response.
    """
    # compute simple features
    distance_km = haversine_km(start_lat, start_lon, end_lat, end_lon)
    # try to call OSRM for route geometry and get number of turns (approx using number of coordinates)
    try:
        coords_str = f"{start_lon},{start_lat};{end_lon},{end_lat}"
        url = f"https://router.project-osrm.org/route/v1/driving/{coords_str}?overview=full&geometries=geojson"
        resp = requests.get(url, timeout=12)
        resp.raise_for_status()
        data = resp.json()
        route = data["routes"][0]["geometry"]["coordinates"]
        num_turns = max(1, len(route) // 10)  # naive proxy for complexity
        distance_m = data["routes"][0].get("distance", distance_km*1000)
        duration_s = data["routes"][0].get("duration", distance_km/60*3600)  # fallback assume 60km/h
    except Exception:
        num_turns = 1
        distance_m = distance_km * 1000
        duration_s = distance_km / 60 * 3600

    # assemble features dictionary / vector for model
    features = {
        "start_lat": start_lat,
        "start_lon": start_lon,
        "end_lat": end_lat,
        "end_lon": end_lon,
        "distance_km": distance_km,
        "distance_m": distance_m,
        "duration_s": duration_s,
        "num_turns": num_turns,
        "traffic_level": traffic_level,
        # simple encodings (model should expect this)
        "is_urban": 1 if distance_km < 10 else 0,
    }

    # if model exists, prepare input exactly with MODEL_FEATURES order
    if MODEL and MODEL_FEATURES:
        try:
            import pandas as pd
            # build dataframe row with feature names
            row = {}
            for f in MODEL_FEATURES:
                if f in features:
                    row[f] = features[f]
                else:
                    # fallback: attempt simple mapping for common names
                    if f == "travel_time_min":
                        row[f] = duration_s / 60.0
                    else:
                        row[f] = 0
            X = pd.DataFrame([row], columns=MODEL_FEATURES)
            pred = MODEL.predict(X)[0]
            return {
                "eta_minutes": float(pred),
                "model_used": True,
                "distance_km": round(distance_km, 3),
                "num_turns": int(num_turns)
            }
        except Exception as e:
            print("Model predict error:", e)

    # fallback heuristic
    # base speed depends on terrain & traffic & priority
    base_speed_kmh = 50.0
    if terrain.lower() in ["hilly", "mountain"]:
        base_speed_kmh = 30.0
    if traffic_level >= 3:
        base_speed_kmh *= 0.6
    if priority == "high":
        base_speed_kmh *= 1.15
    eta_hours = distance_km / max(5.0, base_speed_kmh)
    eta_minutes = eta_hours * 60
    # small penalty per turn
    eta_minutes += num_turns * 0.5
    return {
        "eta_minutes": round(float(eta_minutes), 2),
        "model_used": False,
        "distance_km": round(distance_km, 3),
        "num_turns": int(num_turns)
    }


# -------------------------- Dynamic Rerouting -------------------------
def _parse_closure_points(closure_points_str: str):
    """Parse closure points from a semicolon-separated string like 'lat,lon;lat,lon'"""
    pts = []
    if not closure_points_str:
        return pts
    try:
        for part in closure_points_str.split(";"):
            part = part.strip()
            if not part:
                continue
            lat_str, lon_str = part.split(",")
            pts.append((float(lat_str), float(lon_str)))
    except Exception:
        # ignore parse errors and return what we have
        pass
    return pts


def _route_intersects_closure(route_coords, closures, threshold_km=0.2):
    """Return True if any route point is within threshold_km of any closure point."""
    if not closures:
        return False
    for rc in route_coords:
        # rc is [lon, lat] from OSRM geometry or [lat, lon] depending on caller
        # normalize to (lat, lon)
        if len(rc) >= 2:
            lat = rc[1]
            lon = rc[0]
        else:
            continue
        for c in closures:
            d = haversine_km(lat, lon, c[0], c[1])
            if d <= threshold_km:
                return True
    return False


def _score_route(osrm_route, weather: str, terrain: str, closures: list, traffic_level: int, threshold_km: float = 0.2):
    """Compute a score (lower is better) for an OSRM route dict.
    Uses duration as base, applies multipliers for weather/terrain/traffic and large penalty if intersects closures.
    """
    duration_s = osrm_route.get("duration", 0.0)
    distance_m = osrm_route.get("distance", 0.0)

    weather_mult = {
        "clear": 1.0,
        "cloudy": 1.02,
        "fog": 1.1,
        "rain": 1.25,
        "storm": 1.6,
        "snow": 1.5
    }.get(weather.lower(), 1.0)

    terrain_mult = {
        "plain": 1.0,
        "hilly": 1.15,
        "mountain": 1.3
    }.get(terrain.lower(), 1.0)

    # traffic penalty multiplier
    traffic_mult = 1.0 + max(0, (traffic_level - 1)) * 0.15

    base_score = duration_s * weather_mult * terrain_mult * traffic_mult

    # closure penalty
    # OSRM geometry coordinates are lon,lat list
    coords = osrm_route.get("geometry", {}).get("coordinates") if isinstance(osrm_route.get("geometry"), dict) else osrm_route.get("geometry")
    if coords and _route_intersects_closure(coords, closures, threshold_km=threshold_km):
        # heavy penalty to deprioritize this route
        base_score += 1e7

    # small distance-based tie-breaker
    base_score += distance_m * 0.01
    return base_score



@app.get("/dynamic_reroute", response_class=HTMLResponse)
def dynamic_reroute(start_lat: float = Query(...), start_lon: float = Query(...),
                    end_lat: float = Query(...), end_lon: float = Query(...),
                    weather: str = "clear", terrain: str = "plain",
                    closure_points: str = None, traffic_level: int = 1,
                    closure_threshold_km: float = 0.5):
    """
    Replacement dynamic_reroute endpoint (drop-in).
    Usage same as before. This version:
      - draws original route (blue),
      - draws closed route (red, solid + dashed),
      - draws chosen reroute (green),
      - robustly normalizes coordinates coming from OSRM,
      - marks closure points and shows ETA/summary popup.
    """
    try:
        # ---------- helpers ----------
        def parse_closure_points(s: str):
            pts = []
            if not s:
                return pts
            for part in s.split(";"):
                part = part.strip()
                if not part:
                    continue
                try:
                    lat_str, lon_str = part.split(",")
                    pts.append((float(lat_str), float(lon_str)))
                except Exception:
                    # skip malformed pair
                    continue
            return pts

        def normalize_coord_pair(rc):
            """
            Return (lat, lon) as floats regardless of source order.
            OSRM normally returns [lon, lat]. Protect against unexpected orders.
            Heuristic: latitude must be in [-90,90].
            """
            if not isinstance(rc, (list, tuple)) or len(rc) < 2:
                raise ValueError("invalid coord pair")
            a, b = float(rc[0]), float(rc[1])
            # if 'a' looks like lat (in -90..90) and 'b' outside that, assume (lat,lon)
            if -90.0 <= a <= 90.0 and not (-90.0 <= b <= 90.0):
                return (a, b)
            # if 'b' in -90..90 and 'a' outside, assume (lon,lat) -> swap
            if -90.0 <= b <= 90.0 and not (-90.0 <= a <= 90.0):
                return (b, a)
            # otherwise prefer (lat, lon) if 'b' is lon-like but both plausible -> assume (lon,lat) (OSRM)
            # Most OSRM is [lon, lat] so default to (lat=b, lon=a)
            return (b, a)

        def coords_to_latlon_list(coords_list):
            """Given a list of coordinate pairs (OSRM geometry), return [[lat,lon], ...]"""
            out = []
            for rc in coords_list:
                try:
                    lat, lon = normalize_coord_pair(rc)
                    out.append([lat, lon])
                except Exception:
                    continue
            return out

        closures = parse_closure_points(closure_points)

        # ---------- fetch routes (OSRM with alternatives) ----------
        coords_str = f"{start_lon},{start_lat};{end_lon},{end_lat}"
        url = f"https://router.project-osrm.org/route/v1/driving/{coords_str}?overview=full&geometries=geojson&alternatives=true"
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        if "routes" not in data or not data["routes"]:
            return HTMLResponse("<h3>No routes returned from OSRM</h3>")

        # ---------- store original (primary) route ----------
        original_route = data["routes"][0]

        # ---------- scoring (reuse your _score_route semantics) ----------
        scored = []
        for r in data["routes"]:
            score = _score_route(r, weather, terrain, closures, traffic_level, threshold_km=closure_threshold_km)
            scored.append((score, r))
        scored.sort(key=lambda x: x[0])

        # identify closed routes
        closed_routes = []
        for s, r in scored:
            coords = r.get("geometry", {}).get("coordinates", []) if isinstance(r.get("geometry"), dict) else r.get("geometry")
            if coords and _route_intersects_closure(coords, closures, threshold_km=closure_threshold_km):
                closed_routes.append((s, r))

        # pick best non-closed route if available; else fallback to best overall
        best_non_closed = None
        for s, r in scored:
            coords = r.get("geometry", {}).get("coordinates", []) if isinstance(r.get("geometry"), dict) else r.get("geometry")
            if not (coords and _route_intersects_closure(coords, closures, threshold_km=closure_threshold_km)):
                best_non_closed = (s, r)
                break
        if best_non_closed:
            chosen_score, chosen_route = best_non_closed
        else:
            chosen_score, chosen_route = scored[0]

        closed_to_display = closed_routes[0] if closed_routes else None

        # ---------- build folium map ----------
        # chosen route coords (lat,lon)
        chosen_coords_raw = chosen_route.get("geometry", {}).get("coordinates", []) if isinstance(chosen_route.get("geometry"), dict) else chosen_route.get("geometry")
        route_coords_latlon = coords_to_latlon_list(chosen_coords_raw)

        # safe midpoint for centering
        if route_coords_latlon:
            midpoint = route_coords_latlon[len(route_coords_latlon)//2]
        else:
            midpoint = [(start_lat+end_lat)/2, (start_lon+end_lon)/2]

        m = folium.Map(location=midpoint, zoom_start=12, tiles="cartodb positron")

        # draw all alternatives faintly (grey)
        for s, r in scored:
            coords = r.get("geometry", {}).get("coordinates", []) if isinstance(r.get("geometry"), dict) else r.get("geometry")
            alt_latlon = coords_to_latlon_list(coords)
            if alt_latlon:
                folium.PolyLine(alt_latlon, color="#e6e6e6", weight=4, opacity=0.5).add_to(m)

        # draw original baseline route (blue) if available
        try:
            orig_coords = original_route.get("geometry", {}).get("coordinates", []) if isinstance(original_route.get("geometry"), dict) else original_route.get("geometry")
            orig_latlon = coords_to_latlon_list(orig_coords)
            if orig_latlon:
                folium.PolyLine(orig_latlon, color="#2b7cff", weight=5, opacity=0.7, tooltip="Original route (blue)").add_to(m)
        except Exception:
            pass

        # draw closed route (red) if any
        if closed_to_display:
            cs, cr = closed_to_display
            cr_coords = cr.get("geometry", {}).get("coordinates", []) if isinstance(cr.get("geometry"), dict) else cr.get("geometry")
            cr_latlon = coords_to_latlon_list(cr_coords)
            if cr_latlon:
                # solid red
                folium.PolyLine(cr_latlon, color="#ff3b30", weight=6, opacity=0.95, tooltip="Blocked route (red)").add_to(m)
                # dashed thicker overlay
                folium.PolyLine(cr_latlon, color="#ff3b30", weight=10, opacity=0.85, dash_array="10,8").add_to(m)
                # add closure marker at midpoint of closed route
                cmid = cr_latlon[len(cr_latlon)//2]
                folium.Marker(cmid, popup=folium.Popup("Blocked route (red)", max_width=200),
                              icon=folium.Icon(color="darkred", icon="ban", prefix='fa')).add_to(m)

        # draw chosen optimal reroute (green)
        if route_coords_latlon:
            folium.PolyLine(route_coords_latlon, color="#2ecc71", weight=7, opacity=0.95, tooltip="Chosen optimal route (green)").add_to(m)
            mid = route_coords_latlon[len(route_coords_latlon)//2]
            folium.Marker(mid, popup=folium.Popup("Optimal Route (GREEN)", max_width=200),
                          icon=folium.Icon(color="green", icon="check", prefix='fa')).add_to(m)

        # add start/end markers with ETA (recompute simple multiplier)
        duration_sec = chosen_route.get("duration", 0.0)
        weather_mult = {"clear":1.0, "rain":1.25, "storm":1.6}.get(weather.lower(), 1.0)
        terrain_mult = {"plain":1.0, "hilly":1.15, "mountain":1.3}.get(terrain.lower(), 1.0)
        traffic_mult = 1.0 + max(0, (traffic_level - 1)) * 0.15
        est_duration_min = (duration_sec * weather_mult * terrain_mult * traffic_mult) / 60.0

        from datetime import datetime, timedelta
        dep = datetime.now()
        arr = dep + timedelta(minutes=est_duration_min)

        start_popup = f"""
        <div style="font-family: Arial; width: 260px;">
          <h4 style="margin:5px 0;">Start</h4>
          <p style="margin:2px 0;"><b>Departure:</b> {dep.strftime('%Y-%m-%d %H:%M:%S')}</p>
          <p style="margin:2px 0;"><b>Est. Duration:</b> {est_duration_min:.1f} min</p>
          <p style="margin:2px 0;"><b>Est. Arrival:</b> {arr.strftime('%Y-%m-%d %H:%M:%S')}</p>
          <p style="margin:2px 0;"><b>Score:</b> {chosen_score:.2f}</p>
          <p style="margin:2px 0;"><b>Weather:</b> {weather} | <b>Terrain:</b> {terrain} | <b>Traffic:</b> {traffic_level}</p>
        </div>
        """
        folium.Marker([start_lat, start_lon], popup=folium.Popup(start_popup, max_width=300), icon=folium.Icon(color="green")).add_to(m)

        end_popup = f"""
        <div style="font-family: Arial; width: 200px;">
          <h4 style="margin:5px 0;">Destination</h4>
          <p style="margin:2px 0;"><b>Est. Arrival:</b> {arr.strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        """
        folium.Marker([end_lat, end_lon], popup=folium.Popup(end_popup, max_width=250), icon=folium.Icon(color="red")).add_to(m)

        # show closure points
        for c in closures:
            folium.CircleMarker(location=[c[0], c[1]], radius=6, color="black", fill=True, fill_color="black", fill_opacity=0.9, popup="Closed").add_to(m)

        m.add_child(MeasureControl(primary_length_unit='kilometers'))
        m.add_child(Fullscreen(position='topright'))
        MiniMap(toggle_display=True).add_to(m)
        Draw(export=True).add_to(m)

        return HTMLResponse(content=m._repr_html_())

    except Exception as e:
        return HTMLResponse(f"<h3 style='color:red;'>Error in dynamic_reroute: {e}</h3>")
