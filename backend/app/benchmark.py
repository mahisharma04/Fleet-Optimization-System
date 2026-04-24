import time
import osmnx as ox
import networkx as nx
import numpy as np
import torch
import sys
import os

# Ensure the app directory is in the path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from routing.astar import AStarRouter
from rl.train_rl import PolicyNetwork
from rl.ambulance_env import AmbulanceFleetEnv

def benchmark():
    """Main benchmarking function to compare A* and RL dispatching."""
    print("Starting Algorithm Benchmark...")
    print("-" * 40)
    
    # 1. Load the road network
    try:
        graph_path = "data/raw/pune_network.graphml"
        G = ox.load_graphml(graph_path)
        print("Road network loaded successfully.")
    except FileNotFoundError:
        print("Error: pune_network.graphml not found. Run ingestion.py first.")
        return

    # 2. Setup A* Router
    router = AStarRouter(G)
    
    # 3. Setup RL Policy
    env = AmbulanceFleetEnv(G)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    policy = PolicyNetwork(state_dim, action_dim)
    
    policy_path = "data/processed/ambulance_policy.pth"
    if os.path.exists(policy_path):
        policy.load_state_dict(torch.load(policy_path))
        policy.eval()
        print("RL policy loaded successfully.")
    else:
        print("RL policy not found. Using untrained policy for comparison.")

    # 4. Prepare Test Data (100 random dispatch scenarios)
    num_tests = 100
    nodes = list(G.nodes())
    test_scenarios = []
    for _ in range(num_tests):
        amb_locations = np.random.choice(nodes, 5)
        emergency_loc = np.random.choice(nodes)
        test_scenarios.append((amb_locations, emergency_loc))

    print(f"Running {num_tests} test iterations...")
    print("-" * 40)

    # --- Benchmark A* Algorithm ---
    # A* strategy: Calculate ETA for every idle ambulance and pick the minimum
    start_time = time.time()
    astar_total_response_time = 0
    astar_success_count = 0
    
    for amb_locs, emer_loc in test_scenarios:
        best_eta = float('inf')
        found_path = False
        
        for loc in amb_locs:
            # We use nx.shortest_path_length as a proxy for the A* dispatch decision time
            try:
                # In AStarRouter, we'd find the best route. For benchmarking dispatch speed:
                t = nx.shortest_path_length(G, loc, emer_loc, weight='travel_time')
                if t < best_eta:
                    best_eta = t
                found_path = True
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
        
        if found_path:
            astar_total_response_time += best_eta
            astar_success_count += 1

    astar_exec_time = time.time() - start_time
    astar_avg_resp = astar_total_response_time / astar_success_count if astar_success_count > 0 else 0

    # --- Benchmark RL Algorithm ---
    # RL strategy: Feed state into neural network and get action (ambulance index)
    start_time = time.time()
    rl_total_response_time = 0
    rl_success_count = 0
    
    for amb_locs, emer_loc in test_scenarios:
        # Construct state vector: [loc1, status1, ..., loc5, status5, emer_loc, emer_status]
        state = []
        for loc in amb_locs:
            state.extend([float(loc), 0.0]) # All idle for benchmark
        state.extend([float(emer_loc), 1.0]) # Active emergency
        
        state_tensor = torch.FloatTensor(state)
        with torch.no_grad():
            probs = policy(state_tensor)
            action = torch.argmax(probs).item() # Pick best ambulance
        
        # Calculate actual response time for the chosen ambulance
        try:
            chosen_loc = amb_locs[action]
            resp_time = nx.shortest_path_length(G, chosen_loc, emer_loc, weight='travel_time')
            rl_total_response_time += resp_time
            rl_success_count += 1
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue

    rl_exec_time = time.time() - start_time
    rl_avg_resp = rl_total_response_time / rl_success_count if rl_success_count > 0 else 0

    # 5. Print Results Table
    print(f"{'Metric':<25} | {'A* Search':<15} | {'RL Policy':<15}")
    print("-" * 60)
    print(f"{'Total Exec Time (s)':<25} | {astar_exec_time:<15.4f} | {rl_exec_time:<15.4f}")
    print(f"{'Avg Exec Per Call (ms)':<25} | {(astar_exec_time/num_tests)*1000:<15.4f} | {(rl_exec_time/num_tests)*1000:<15.4f}")
    print(f"{'Avg Response Time (s)':<25} | {astar_avg_resp:<15.2f} | {rl_avg_resp:<15.2f}")
    print(f"{'Success Rate (%)':<25} | {(astar_success_count/num_tests)*100:<15.1f} | {(rl_success_count/num_tests)*100:<15.1f}")
    print("-" * 60)
    
    print("Observations:")
    print("1. RL is typically faster for decision-making as it involves a single forward pass.")
    print("2. A* is mathematically optimal for pathfinding but slower when checking multiple candidates.")
    print("3. Response times depend on RL training quality.")

def main():
    """Entry point for the benchmark script."""
    # Perform benchmarking
    benchmark()

if __name__ == "__main__":
    main()
