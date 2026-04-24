import requests
import time
import random
import numpy as np

# API Configuration
BASE_URL = "http://localhost:8000"
EMERGENCY_ENDPOINT = f"{BASE_URL}/emergency"

# Pune area bounds (approximate for simulation)
LAT_RANGE = (18.51, 18.53)
LON_RANGE = (73.84, 73.87)

PRIORITIES = ["High", "Medium", "Low"]

def generate_random_emergency():
    """Generate a random emergency location and priority."""
    lat = random.uniform(*LAT_RANGE)
    lon = random.uniform(*LON_RANGE)
    priority = random.choice(PRIORITIES)
    
    return {
        "location": [lat, lon],
        "priority": priority
    }

def run_simulation(interval=10, duration=None):
    """
    Simulate an active system by automatically injecting emergencies.
    
    Args:
        interval: Seconds between new emergency calls.
        duration: Total seconds to run the simulation (None for infinite).
    """
    print("Starting System Simulation...")
    print(f"Injecting a new emergency every {interval} seconds.")
    print("Press Ctrl+C to stop.")
    print("-" * 40)
    
    start_time = time.time()
    count = 0
    
    try:
        while True:
            # Check duration
            if duration and (time.time() - start_time) > duration:
                break
                
            # Create emergency data
            payload = generate_random_emergency()
            
            try:
                # Send POST request to the API
                response = requests.post(EMERGENCY_ENDPOINT, json=payload)
                
                if response.status_code == 200:
                    data = response.json()
                    count += 1
                    status = data.get("status", "unknown")
                    req_id = data.get("request_id", "N/A")
                    
                    print(f"[{time.strftime('%H:%M:%S')}] Created Emergency #{req_id}")
                    print(f"  Priority: {payload['priority']}")
                    print(f"  Location: {payload['location']}")
                    print(f"  Status: {status}")
                    
                    if status == "dispatched":
                        print(f"  Assigned Ambulance: #{data.get('assigned_ambulance')}")
                        print(f"  ETA: {data.get('eta'):.2f}s")
                    else:
                        print(f"  Message: {data.get('message')}")
                    print("-" * 20)
                else:
                    print(f"Error: Received status code {response.status_code}")
                    
            except requests.exceptions.ConnectionError:
                print("Error: Could not connect to the backend. Is main.py running?")
            
            # Wait for the next interval
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")
        
    print("-" * 40)
    print(f"Simulation finished. Total emergencies created: {count}")

def main():
    """Main entry point for the simulation script."""
    # Run the simulation: 1 emergency every 10 seconds
    run_simulation(interval=10)

if __name__ == "__main__":
    main()
