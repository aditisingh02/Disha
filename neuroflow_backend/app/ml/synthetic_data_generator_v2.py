
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from pathlib import Path
import logging

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TrafficGenerator")

class TrafficDataGenerator:
    def __init__(self, output_dir="neuroflow_backend/data/datasets_v2"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Reality Anchors: City-Specific Profiles
        self.city_profiles = {
            "mumbai": {
                "base_speed": 40,
                "peak_speed": 15,
                "segments": 300,
                "weather_bias": "monsoon", # Heavy rain impact
                "events": ["Ganesh Chaturthi", "Cricket Match at Wankhede", "Bollywood Premiere"]
            },
            "bengaluru": {
                "base_speed": 35,
                "peak_speed": 5, # Gridlock
                "segments": 250,
                "weather_bias": "pleasant",
                "events": ["Tech Summit", "IPL Match at Chinnaswamy", "Start-up Expo"]
            },
            "delhi": {
                "base_speed": 50, # Wide roads
                "peak_speed": 20,
                "segments": 350,
                "weather_bias": "extreme", # Fog/Heat
                "events": ["Republic Day", "Trade Fair", "Political Rally"]
            }
        }

    def generate_time_series(self, start_date, days=30):
        """Generate 15-min interval timestamps"""
        end_date = start_date + timedelta(days=days)
        return pd.date_range(start=start_date, end=end_date, freq="15min")

    def get_weather_impact(self, city, date, month):
        """
        Simulate weather based on city and season.
        Returns: (is_rainy, rain_intensity, visibility_impact_factor)
        """
        is_rainy = False
        rain_intensity = 0.0
        impact = 1.0 # 1.0 = No impact
        
        if city == "mumbai":
            if month in [6, 7, 8, 9]: # Monsoon
                if random.random() < 0.6: # 60% chance of rain in monsoon
                    is_rainy = True
                    rain_intensity = random.uniform(0.5, 1.0) # High intensity
                    impact = 0.6 # 40% speed drop
            elif random.random() < 0.05: # Random shower
                is_rainy = True
                rain_intensity = random.uniform(0.1, 0.4)
                impact = 0.9

        elif city == "bengaluru":
            if month in [5, 6, 9, 10]:
                if random.random() < 0.3:
                    is_rainy = True
                    rain_intensity = random.uniform(0.2, 0.6)
                    impact = 0.8
        
        elif city == "delhi":
            # Winter Fog Logic
            if month in [12, 1]: 
                if random.random() < 0.7: # High fog chance
                    impact = 0.7 # Visibility drop
            # Summer Heat
            elif month in [5, 6]:
                impact = 0.95 # Slight slowdown due to heat?

        return is_rainy, rain_intensity, impact

    def generate_city_data(self, city_name, start_date_str="2026-01-01"):
        logger.info(f"Generating rich data for {city_name}...")
        
        profile = self.city_profiles[city_name]
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
        timestamps = self.generate_time_series(start_date, days=45) # 45 days data
        
        data = []
        
        # Pre-compute event calendar
        event_days = {}
        for day in pd.date_range(start=start_date, end=start_date + timedelta(days=45), freq="D"):
            if random.random() < 0.15: # 15% chance of an event
                event_days[day.date()] = {
                    "nearby_events": random.choice(profile["events"]),
                    "event_attendance": random.randint(5000, 50000),
                    "has_major_event": True
                }
        
        # Road Network Simulation
        segments = [f"{city_name}_seg_{i}" for i in range(profile["segments"])]
        road_categories = ["Arterial", "Highway", "Collector", "Local"]
        segment_props = {
            s: {
                "road_name": f"Road {random.randint(1, 99)}",
                "road_category": random.choice(road_categories),
                "lat": 19.0760 + random.uniform(-0.1, 0.1) if city_name == "mumbai" else (12.9716 + random.uniform(-0.1, 0.1) if city_name == "bengaluru" else 28.7041 + random.uniform(-0.1, 0.1)),
                "lon": 72.8777 + random.uniform(-0.1, 0.1) if city_name == "mumbai" else (77.5946 + random.uniform(-0.1, 0.1) if city_name == "bengaluru" else 77.1025 + random.uniform(-0.1, 0.1)),
                "base_speed_mod": random.uniform(0.8, 1.2)
            } for s in segments
        }

        for ts in timestamps:
            # Temporal Features
            hour = ts.hour
            day_of_week = ts.dayofweek
            is_weekend = day_of_week >= 5
            is_peak = (8 <= hour <= 11) or (17 <= hour <= 20)
            month = ts.month
            date = ts.date()
            
            # Weather
            is_rainy, rain_intensity, weather_impact = self.get_weather_impact(city_name, date, month)
            
            # Event
            event_info = event_days.get(date, {
                "nearby_events": "None",
                "event_attendance": 0,
                "has_major_event": False
            })
            
            # Event Impact
            event_impact = 1.0
            if event_info["has_major_event"] and (16 <= hour <= 22): # Evening events
                event_impact = 0.7 # Congestion near event
            
            # Incident Generation (Random accidents)
            
            for seg_id in segments:
                props = segment_props[seg_id]
                has_incident = random.random() < 0.001 # 0.1% chance per 15 min
                incident_impact = 0.3 if has_incident else 1.0
                
                # Base Traffic Pattern (Sine wave for daily cycle)
                # Peak at 9 AM and 6 PM
                daily_pattern = np.sin((hour - 6) * np.pi / 12) * 0.5 + 0.5 # 0 to 1
                
                if is_peak and not is_weekend:
                    congestion_factor = 0.4 # Slow
                elif is_peak and is_weekend:
                    congestion_factor = 0.7 # Moderate
                elif 1 <= hour <= 4:
                    congestion_factor = 1.5 # Fast (Night)
                else:
                    congestion_factor = 1.0
                
                # Final Speed Calculation
                # Speed = Base * Factors
                # Factors: Weather * Event * Incident * Congestion
                
                raw_speed = profile["base_speed"] * props["base_speed_mod"] * \
                            congestion_factor * weather_impact * event_impact * incident_impact
                
                # Add Noise
                speed = max(5, min(120, raw_speed + random.gauss(0, 2)))
                
                # Derived Metrics
                volume = int((120 - speed) * 15 * random.uniform(0.8, 1.2)) # Inverse to speed
                occupancy = min(1.0, volume / 2000)
                
                row = {
                    "source": "neuroflow_synthetic_v2",
                    "timestamp": ts,
                    "hour": hour,
                    "day_of_week": day_of_week,
                    "is_weekday": not is_weekend,
                    "is_peak_hour": is_peak,
                    "road_name": props["road_name"],
                    "road_category": props["road_category"],
                    "latitude": props["lat"],
                    "longitude": props["lon"],
                    "city": city_name,
                    "segment_id": seg_id,
                    "speed": round(speed, 2),
                    "volume": volume,
                    "occupancy": round(occupancy, 4),
                    "speed_band": "High" if speed > 60 else ("Medium" if speed > 30 else "Low"),
                    "has_incident": has_incident,
                    "month": month,
                    "nearby_events": event_info["nearby_events"],
                    "event_attendance": event_info["event_attendance"],
                    "has_major_event": event_info["has_major_event"],
                    # New Columns
                    "event_max_rank": random.randint(1, 10) if event_info["has_major_event"] else 0,
                    "has_sports_event": "Match" in str(event_info["nearby_events"]),
                    "has_concert": "Premiere" in str(event_info["nearby_events"]),
                    "has_conference": "Summit" in str(event_info["nearby_events"]),
                    "date": date,
                    "is_rainy": is_rainy,
                    "rain_intensity": round(rain_intensity, 2),
                    "extreme_weather_flag": weather_impact < 0.7,
                    "weather_severity_index": round((1-weather_impact)*10, 1),
                    # Lags (Simplified for synthetic: just mock previous values roughly)
                    "precipitation_lag1": round(rain_intensity * 0.9, 2), 
                    "precipitation_lag3": round(rain_intensity * 0.7, 2),
                    "precipitation_lag6": round(rain_intensity * 0.5, 2),
                    "is_holiday": is_weekend, 
                    "holiday_type": "Weekend" if is_weekend else "None",
                    "holiday_intensity_score": 0.8 if is_weekend else 0.2,
                    "peak_day_probability": 0.9 if not is_weekend else 0.3,
                    "peak_driver_type": "Commuter" if not is_weekend else "Leisure"
                }
                data.append(row)
                
        # Convert to DataFrame
        df = pd.DataFrame(data)
        output_path = self.output_dir / f"{city_name}_train_v2.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(df)} rows to {output_path}")

if __name__ == "__main__":
    generator = TrafficDataGenerator()
    generator.generate_city_data("mumbai")
    generator.generate_city_data("bengaluru")
    generator.generate_city_data("delhi")
