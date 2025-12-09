# generate_synthetic_data.py
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

N = 5000
rows = []
base = datetime(2024,1,1,0,0,0)
for i in range(N):
    start = base + timedelta(minutes=random.randint(0, 60*24*365))
    duration_s = abs(int(np.random.normal(loc=3600, scale=1200)))  # 1 hour avg
    end = start + timedelta(seconds=duration_s)
    # random coords roughly in Delhi region
    slat = 28.5 + random.random()*0.5
    slon = 77.0 + random.random()*0.5
    elat = 28.4 + random.random()*0.7
    elon = 76.9 + random.random()*0.7
    distance_m = max(1000, int(np.random.normal(loc=40000, scale=15000)))  # 40 km avg
    avg_speed_kmh = (distance_m/1000) / (duration_s/3600)
    traffic_level = random.choices([0,1,2,3], weights=[0.2,0.4,0.3,0.1])[0]
    weather = random.choices(['clear','rain','fog','storm'], weights=[0.7,0.18,0.07,0.05])[0]
    terrain = random.choices(['flat','hilly','mountainous'], weights=[0.7,0.25,0.05])[0]
    vehicle_type = random.choice(['truck','troop','tank'])
    priority = random.choices(['low','med','high'], weights=[0.6,0.3,0.1])[0]
    # add noise to duration with traffic, weather, terrain, priority
    delay_factor = 1 + traffic_level*0.12 + (0.15 if weather=='rain' else 0) + (0.2 if terrain!='flat' else 0)
    if priority=='high': delay_factor *= 0.85
    duration_s = int((distance_m/1000) / max(5, avg_speed_kmh) * 3600 * delay_factor)
    rows.append({
        'trip_id': f"trip_{i}",
        'start_ts': start.isoformat(),
        'end_ts': end.isoformat(),
        'start_lat': slat,
        'start_lon': slon,
        'end_lat': elat,
        'end_lon': elon,
        'distance_m': distance_m,
        'duration_s': duration_s,
        'traffic_level': traffic_level,
        'weather': weather,
        'terrain': terrain,
        'vehicle_type': vehicle_type,
        'priority': priority
    })

df = pd.DataFrame(rows)
df.to_csv('synthetic_trips.csv', index=False)
print("wrote synthetic_trips.csv", df.shape)
