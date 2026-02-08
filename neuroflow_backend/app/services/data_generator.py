"""
NeuroFlow BharatFlow — Multi-City Synthetic Traffic Data Generator
Acts as a "Digital Twin" for Bengaluru, Mumbai, and Delhi.
Generates realistic traffic patterns, weather events, and special incidents.
"""

import asyncio
import logging
import random
import csv
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np

from app.core.config import settings

logger = logging.getLogger("neuroflow.data_generator")

# ── City Configurations ──
CITY_CONFIGS = {
    "singapore": {
        "center": {"lat": 1.3521, "lng": 103.8198},
        "radius_km": 20,
        "segments": 100,
        "patterns": {
            "morning_rush_start": 7, "morning_rush_end": 10,
            "evening_rush_start": 17, "evening_rush_end": 20,
            "rain_prob": 0.5, "rain_impact": 0.30,
        }
    },
    "bengaluru": {
        "center": {"lat": 12.9716, "lng": 77.5946},
        "radius_km": 15,
        "segments": 250,
        "patterns": {
            "morning_rush_start": 8, "morning_rush_end": 11,
            "evening_rush_start": 17, "evening_rush_end": 21,
            "rain_prob": 0.2, "rain_impact": 0.35,  # 35% slowdown
        }
    },
    "mumbai": {
        "center": {"lat": 19.0760, "lng": 72.8777},
        "radius_km": 20,
        "segments": 300,
        "patterns": {
            "morning_rush_start": 8, "morning_rush_end": 11,
            "evening_rush_start": 18, "evening_rush_end": 22, # Late evening rush
            "rain_prob": 0.4, "rain_impact": 0.50,  # Severe flooding impact
        }
    },
    "delhi": {
        "center": {"lat": 28.6139, "lng": 77.2090},
        "radius_km": 25,
        "segments": 350,
        "patterns": {
            "morning_rush_start": 8, "morning_rush_end": 10,
            "evening_rush_start": 17, "evening_rush_end": 20,
            "rain_prob": 0.1, "rain_impact": 0.20,
            "fog_prob": 0.3, "fog_impact": 0.40, # Winter smog/fog
        }
    }
}

class TrafficDataGenerator:
    def __init__(self, output_dir: str = "data/datasets"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_all_cities(self, days: int = 30):
        """Generate datasets for all configured cities."""
        for city_name in CITY_CONFIGS.keys():
            logger.info(f"Generating {days} days of data for {city_name}...")
            self.generate_city_data(city_name, days)
        logger.info("✅ All city datasets generated successfully.")

    def generate_city_data(self, city_name: str, days: int):
        config = CITY_CONFIGS[city_name]
        segments = self._generate_topology(city_name, config)
        
        filename = self.output_dir / f"{city_name}_train.csv"
        
        start_time = datetime.now() - timedelta(days=days)
        interval = timedelta(minutes=5)
        total_steps = int((days * 24 * 60) / 5)

        logger.info(f"  - Simulating {total_steps} timesteps for {len(segments)} segments...")

        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ['timestamp', 'city', 'segment_id', 'speed', 'volume', 'occupancy', 'weather', 'is_holiday', 'special_event']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            current_time = start_time
            for _ in range(total_steps):
                # Sim World State
                is_weekend = current_time.weekday() >= 5
                hour = current_time.hour
                
                # Weather & Event Logic
                weather, weather_impact = self._get_weather(config, current_time)
                event, event_impact = self._get_special_event(city_name, current_time)

                # Rush Hour Logic relative to city
                rush_factor = self._get_rush_factor(hour, is_weekend, config["patterns"])

                for seg in segments:
                    # Base Physics
                    base_speed = seg['base_speed']
                    capacity = seg['capacity']

                    # Apply Impacts
                    # Speed = Base * (1 - Rush) * (1 - Weather) * (1 - Event)
                    congestion = rush_factor * (1.0 + np.random.uniform(-0.1, 0.1))
                    actual_speed = base_speed * (1.0 - congestion) * (1.0 - weather_impact) * (1.0 - event_impact)
                    actual_speed = max(2.0, actual_speed) # Minimum 2km/h (gridlock)

                    # Sim derived metrics
                    # Volume follows congestion (inverted U-shape in reality, but linear approx for sensors)
                    volume = int(capacity * congestion * 0.8) 
                    occupancy = min(0.95, congestion + (0.1 if weather != "Clear" else 0))

                    writer.writerow({
                        'timestamp': current_time.isoformat(),
                        'city': city_name,
                        'segment_id': seg['id'],
                        'speed': round(actual_speed, 2),
                        'volume': volume,
                        'occupancy': round(occupancy, 3),
                        'weather': weather,
                        'is_holiday': is_weekend,
                        'special_event': event
                    })

                current_time += interval

        logger.info(f"  - Saved to {filename}")

    def _generate_topology(self, city: str, config: Dict) -> List[Dict]:
        """Creates a mock graph of road segments for the city."""
        segments = []
        center = config["center"]
        for i in range(config["segments"]):
            # Random spread around center
            lat = center["lat"] + np.random.uniform(-0.05, 0.05)
            lng = center["lng"] + np.random.uniform(-0.05, 0.05)
            
            # Physics properties
            road_type = np.random.choice(["highway", "arterial", "local"], p=[0.1, 0.3, 0.6])
            if road_type == "highway":
                base_speed = 80
                capacity = 4000
            elif road_type == "arterial":
                base_speed = 50
                capacity = 2000
            else:
                base_speed = 30
                capacity = 800

            segments.append({
                "id": f"{city}_seg_{i}",
                "lat": lat,
                "lng": lng,
                "base_speed": base_speed,
                "capacity": capacity
            })
        return segments

    def _get_rush_factor(self, hour: int, is_weekend: bool, patterns: Dict) -> float:
        """Returns 0.0 (empty) to 1.0 (gridlock)."""
        if is_weekend:
            # Weekend pattern: Mid-day shopping rush
            if 11 <= hour <= 19: return 0.4
            return 0.1

        # Weekday
        if patterns["morning_rush_start"] <= hour < patterns["morning_rush_end"]:
            return 0.85 # Peak Morning
        if patterns["evening_rush_start"] <= hour < patterns["evening_rush_end"]:
            return 0.90 # Peak Evening (Usually worse)
        if 23 <= hour or hour < 5:
            return 0.05 # Night
        return 0.3 # Mid-day

    def _get_weather(self, config: Dict, dt: datetime) -> Tuple[str, float]:
        """Returns (Weather Condition, Impact Factor)."""
        # Monsoon Season: June to Sept
        is_monsoon = 6 <= dt.month <= 9
        if is_monsoon and random.random() < config["patterns"]["rain_prob"]:
            return "Rain", config["patterns"]["rain_impact"]
        
        # Fog in Winter (Delhi specific)
        if "fog_prob" in config["patterns"] and (12 <= dt.month or dt.month <= 1):
             if 5 <= dt.hour <= 9 and random.random() < config["patterns"]["fog_prob"]:
                 return "Fog", config["patterns"]["fog_impact"]

        return "Clear", 0.0

    def _get_special_event(self, city: str, dt: datetime) -> Tuple[str, float]:
        """Simulates events like IPL Matches or Festivals."""
        # Simple probability based events
        if city == "mumbai" and dt.month == 9 and dt.day in [10, 11] and 16 <= dt.hour <= 22:
            return "Ganpati Visarjan", 0.6 # Massive 60% impact
        
        if city == "bengaluru" and dt.weekday() == 6 and 18 <= dt.hour <= 22:
             if random.random() < 0.1: # Occasional Sunday Match
                 return "IPL Match", 0.4

        return "None", 0.0

if __name__ == "__main__":
    # Create simulator and run
    logging.basicConfig(level=logging.INFO)
    gen = TrafficDataGenerator()
    gen.generate_all_cities(days=30)
