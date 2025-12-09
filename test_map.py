import folium
import requests
from folium.plugins import MeasureControl, Fullscreen, MiniMap, Draw
from fastapi.responses import HTMLResponse, FileResponse

@app.get("/visualize_route", response_class=HTMLResponse)
def visualize_route():
    try:
        # Start & End coordinates [latitude, longitude]
        start = [28.6139, 77.2090]   # Delhi
        end = [28.4595, 77.0266]     # Gurugram

        # OSRM public API call
        url = f"http://router.project-osrm.org/route/v1/driving/{start[1]},{start[0]};{end[1]},{end[0]}?overview=full&geometries=geojson"
        response = requests.get(url)
        data = response.json()

        # Extract route geometry
        route = data['routes'][0]['geometry']['coordinates']
        # Convert [lon, lat] -> [lat, lon] for folium
        route = [[coord[1], coord[0]] for coord in route]

        # Create folium map centered mid-route
        midpoint = route[len(route)//2]
        m = folium.Map(location=midpoint, zoom_start=11, tiles="CartoDB positron")

        # Plot the actual road route
        folium.PolyLine(route, color="red", weight=6, opacity=0.8).add_to(m)

        # Add markers
        folium.Marker(start, popup="Start: Delhi 🟢", icon=folium.Icon(color="green")).add_to(m)
        folium.Marker(end, popup="Destination: Gurugram ⚑", icon=folium.Icon(color="red")).add_to(m)

        # Add interactive controls
        m.add_child(MeasureControl(primary_length_unit='kilometers'))
        m.add_child(Fullscreen(position='topright'))
        MiniMap(toggle_display=True).add_to(m)
        Draw(export=True).add_to(m)

        # Save map
        map_path = "route_map.html"
        m.save(map_path)
        return FileResponse(map_path)
    except Exception as e:
        return {"error": str(e)}



