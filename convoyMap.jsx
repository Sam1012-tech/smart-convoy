import { MapContainer, TileLayer, Polyline, Marker, Circle, Popup } from "react-leaflet";
import { useEffect, useState, useRef } from "react";

export default function ConvoyMap() {
  const [route, setRoute] = useState([]);
  const [originalRoute, setOriginalRoute] = useState([]);
  const [closures, setClosures] = useState([]);
  const [closedSegments, setClosedSegments] = useState([]);
  const [convoyPos, setConvoyPos] = useState(null);
  const mapRef = useRef();

  // Start → End (you can make this user input)
  const start = { lat: 19.0760, lon: 72.8777 };
  const end   = { lat: 19.2183, lon: 72.9781 };

  // ---------------------------
  // 1) Fetch initial route
  // ---------------------------
  useEffect(() => {
    const url = `http://127.0.0.1:8000/dynamic_reroute_json?start_lat=${start.lat}&start_lon=${start.lon}&end_lat=${end.lat}&end_lon=${end.lon}`;

    fetch(url)
      .then(res => res.json())
      .then(data => {
        setOriginalRoute(data.original_route);
        setRoute(data.chosen_route);
        setClosures(data.closures || []);
        setClosedSegments(data.closed_segments || []);
        if (data.chosen_route.length > 0) {
          setConvoyPos(data.chosen_route[0]); // starting point
        }
      });
  }, []);

  // ---------------------------
  // 2) WebSocket — Live updates
  // ---------------------------
  useEffect(() => {
    const ws = new WebSocket("ws://127.0.0.1:8000/ws");

    ws.onopen = () => {
      ws.send(
        JSON.stringify({
          start_lat: start.lat,
          start_lon: start.lon,
          end_lat: end.lat,
          end_lon: end.lon,
          closure_points: ""
        })
      );
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);

      setRoute(data.route);
      setConvoyPos(data.convoy_pos);
      setClosedSegments(data.closed_segments);
    };

    return () => ws.close();
  }, []);

  return (
    <MapContainer
      center={[start.lat, start.lon]}
      zoom={12}
      ref={mapRef}
      style={{ height: "100vh", width: "100%" }}
    >
      {/* Basemap */}
      <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />

      {/* Original Route (Blue) */}
      <Polyline positions={originalRoute} pathOptions={{ color: "blue" }} />

      {/* Safe Chosen Route (Green) */}
      <Polyline positions={route} pathOptions={{ color: "green" }} />

      {/* Closures (Red circles) */}
      {closures.map((c, i) => (
        <Circle key={i} center={c} radius={300} pathOptions={{ color: "red" }} />
      ))}

      {/* Closed Segments (Red dots) */}
      {closedSegments.map((c, i) => (
        <Marker key={i} position={c}>
          <Popup>Segment avoided due to hazard</Popup>
        </Marker>
      ))}

      {/* Live Convoy Position (Orange Marker) */}
      {convoyPos && (
        <Marker position={convoyPos}>
          <Popup>Convoy Position</Popup>
        </Marker>
      )}
    </MapContainer>
  );
}
