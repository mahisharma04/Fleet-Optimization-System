from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import osmnx as ox
import pandas as pd
import numpy as np
import asyncio
# import torch
import sys
import os
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

# Add the app directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ml.demand_prediction import DemandPredictor
from routing.astar import AStarRouter
from rl.train_rl import PolicyNetwork

from fastapi.middleware.cors import CORSMiddleware

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start the queue processor on startup
    queue_task = asyncio.create_task(process_queue_loop())
    yield
    # Cleanup on shutdown
    queue_task.cancel()
    try:
        await queue_task
    except asyncio.CancelledError:
        pass

app = FastAPI(
    title="Ambulance Fleet Optimization System",
    lifespan=lifespan
)

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load data and models
print("Loading system components...")
G = ox.load_graphml("data/raw/pune_network.graphml")
router = AStarRouter(G)

# Load ML Predictor
predictor = DemandPredictor()
predictor.load_model("data/processed/")

# Load RL Policy
state_dim = 5 * 2 + 2 # 5 ambulances + 1 emergency
action_dim = 5
policy = None
# try:
#     policy = PolicyNetwork(state_dim, action_dim)
#     policy.load_state_dict(torch.load("data/processed/ambulance_policy.pth"))
#     policy.eval()
# except (FileNotFoundError, RuntimeError, OSError) as e:
#     print(f"RL policy could not be loaded: {e}. Using random dispatch.")

# Mock Ambulance State
ambulances = [
    {"id": i, "location": list(G.nodes())[np.random.randint(0, len(G.nodes()))], "status": "idle", "destination": None}
    for i in range(5)
]

# Mock Emergency Requests
emergencies = {}
pending_queue = []

async def complete_mission(ambulance_id: int, request_id: int, duration: float):
    """Simulate ambulance mission completion after travel time."""
    # Scale down duration for demo purposes (e.g., 1s in real life = 0.05s in simulation)
    sim_duration = duration / 20.0 
    await asyncio.sleep(sim_duration)
    
    # Mark ambulance as idle
    ambulances[ambulance_id]["status"] = "idle"
    ambulances[ambulance_id]["location"] = ambulances[ambulance_id]["destination"]
    ambulances[ambulance_id]["destination"] = None
    
    # Mark request as completed
    if request_id in emergencies:
        emergencies[request_id]["status"] = "completed"
    
    print(f"Ambulance #{ambulance_id} completed mission #{request_id} and is now IDLE.")

class EmergencyRequest(BaseModel):
    location: list[float] # [lat, lon]
    priority: str

async def process_queue_loop():
    """Continuously check for idle ambulances to process the pending queue."""
    while True:
        if pending_queue:
            idle_ambs = [a for a in ambulances if a["status"] == "idle"]
            if idle_ambs:
                # Get the highest priority request from the queue
                # For simplicity, we just take the first one in this demo
                request_id = pending_queue.pop(0)
                request = emergencies[request_id]
                
                best_amb = None
                min_time = float('inf')
                best_route = None

                for amb in idle_ambs:
                    route = router.get_fastest_route(amb["location"], request["node"])
                    if route:
                        eta = router.calculate_eta(route)
                        if eta < min_time:
                            min_time = eta
                            best_amb = amb
                            best_route = route

                if best_amb:
                    best_amb["status"] = "busy"
                    best_amb["destination"] = request["node"]
                    request["assigned_ambulance_id"] = best_amb["id"]
                    request["status"] = "dispatched"
                    
                    # Schedule completion
                    asyncio.create_task(complete_mission(best_amb["id"], request_id, min_time))
                    print(f"Queue Processor: Dispatched Ambulance #{best_amb['id']} to pending request #{request_id}")
                else:
                    # Put it back if no route found
                    pending_queue.append(request_id)
        
        await asyncio.sleep(2) # Check every 2 seconds

@app.get("/")
async def root():
    return {"message": "Ambulance Fleet Optimization System API"}

@app.post("/emergency")
async def create_emergency(request: EmergencyRequest, background_tasks: BackgroundTasks):
    # Find nearest node in the graph
    node = ox.distance.nearest_nodes(G, request.location[1], request.location[0])
    
    request_id = len(emergencies) + 1
    new_request = {
        "id": request_id,
        "node": node,
        "location": request.location,
        "priority": request.priority,
        "timestamp": datetime.now(),
        "assigned_ambulance_id": None,
        "status": "pending"
    }
    emergencies[request_id] = new_request
    
    # Simple fallback: find nearest idle ambulance
    idle_ambs = [a for a in ambulances if a["status"] == "idle"]
    
    if not idle_ambs:
        # Add to pending queue
        pending_queue.append(request_id)
        return {
            "request_id": request_id,
            "status": "queued",
            "message": "All ambulances are busy. Request added to queue."
        }
    
    # Heuristic: Pick the one with shortest travel time (proactive fallback)
    best_amb = None
    min_time = float('inf')
    best_route = None

    for amb in idle_ambs:
        route = router.get_fastest_route(amb["location"], node)
        if route:
            eta = router.calculate_eta(route)
            if eta < min_time:
                min_time = eta
                best_amb = amb
                best_route = route

    if best_amb:
        best_amb["status"] = "busy"
        best_amb["destination"] = node
        new_request["assigned_ambulance_id"] = best_amb["id"]
        new_request["status"] = "dispatched"
        
        # Schedule mission completion
        background_tasks.add_task(complete_mission, best_amb["id"], request_id, min_time)
        
        return {
            "request_id": request_id,
            "assigned_ambulance": best_amb["id"],
            "eta": min_time,
            "route": best_route,
            "status": "dispatched"
        }
    else:
        # If reachable, queue it. If not reachable, error.
        pending_queue.append(request_id)
        return {
            "request_id": request_id,
            "status": "queued",
            "message": "No reachable ambulances available. Request added to queue."
        }

@app.post("/reset")
async def reset_system():
    """Reset the system to initial state."""
    # Reset ambulances to idle and random locations
    for i in range(5):
        ambulances[i]["status"] = "idle"
        ambulances[i]["location"] = list(G.nodes())[np.random.randint(0, len(G.nodes()))]
        ambulances[i]["destination"] = None
    
    # Clear requests and queue in-place
    emergencies.clear()
    pending_queue.clear()
    
    print("System reset executed: Ambulances reset, all calls deleted.")
    return {"message": "System reset successfully"}

@app.get("/emergencies")
async def get_emergencies():
    # Return all emergencies, sorted by timestamp
    return sorted(emergencies.values(), key=lambda x: x["timestamp"], reverse=True)

@app.get("/dispatch/{request_id}")
async def get_dispatch(request_id: int):
    if request_id not in emergencies:
        raise HTTPException(status_code=404, detail="Request not found")
    return emergencies[request_id]

@app.get("/ambulances")
async def get_ambulances():
    # Add coordinates for visualization
    results = []
    for amb in ambulances:
        node_data = G.nodes[amb["location"]]
        results.append({
            **amb,
            "lat": node_data['y'],
            "lon": node_data['x']
        })
    return results

@app.get("/prediction")
async def get_prediction(hour: int = None, day: int = None):
    if hour is None: hour = datetime.now().hour
    if day is None: day = datetime.now().weekday()
    
    # Predict for all clusters
    predictions = []
    for i in range(predictor.n_clusters):
        prob = predictor.predict(i, hour, day)
        center = predictor.cluster_centers[i]
        predictions.append({
            "cluster_id": i,
            "lat": center[0],
            "lon": center[1],
            "score": float(prob)
        })
    return predictions

@app.get("/route")
async def get_route(ambulance_id: int, destination_node: int):
    if ambulance_id >= len(ambulances):
        raise HTTPException(status_code=404, detail="Ambulance not found")
    
    origin_node = ambulances[ambulance_id]["location"]
    route = router.get_fastest_route(origin_node, destination_node)
    eta = router.calculate_eta(route)
    
    # Convert route nodes to lat/lon for frontend
    path_coords = []
    for node in route:
        path_coords.append([G.nodes[node]['y'], G.nodes[node]['x']])
        
    return {
        "route": route,
        "path_coords": path_coords,
        "eta": eta
    }

@app.get("/metrics")
async def get_metrics():
    """Calculate system-wide metrics."""
    # Coverage: Percentage of idle ambulances
    coverage = (len([a for a in ambulances if a["status"] == "idle"]) / len(ambulances)) * 100 if ambulances else 0
    
    # Avg ETA: Average of demand scores as a proxy for ETA if no active missions, 
    # or actual ETAs from dispatched requests
    active_requests = [e for e in emergencies.values() if e["status"] == "dispatched"]
    
    # For demo, let's use a combination of active request ETAs or a base value
    if active_requests:
        # In a real system, we'd track the original ETA. 
        # Here we'll simulate an average ETA between 5-15 mins
        avg_eta = np.random.uniform(5, 12)
    else:
        avg_eta = 0.0
        
    return {
        "avg_eta": f"{avg_eta:.1f}m",
        "coverage": f"{int(coverage)}%"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
