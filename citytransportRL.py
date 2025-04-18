import gym
import CityTransportEnv
import pickle
import osmnx as ox

# Define the graph and bus_stops
#graph = {}  # Replace with actual graph data
#bus_stops = []  # Replace with actual bus stops data

# Create the environment
#env = CityTransportEnv(graph, bus_stops, num_buses=2)
# Load the graph and bus_stops from a file
# Load the graph
with open("city_graph.pkl", "rb") as f:
    graph = pickle.load(f)

# Load the bus stops
bus_stops = ox.geometries_from_place("San Francisco, California, USA", tags={"highway": "bus_stop"})
bus_stops = bus_stops[bus_stops.geometry.type == 'Point']
# Extract coordinates (latitude and longitude) from geometry

bus_stops['x'] = bus_stops.geometry.x  # Longitude
bus_stops['y'] = bus_stops.geometry.y  # Latitude
# Map each bus stop to the nearest node in the graph
bus_stops['nearest_node'] = bus_stops.apply(
    lambda row: ox.distance.nearest_nodes(graph, X=row['x'], Y=row['y']), axis=1
)

# Save the graph and bus stops as a tuple
with open("city_graph_with_bs.pkl", "wb") as f:
    pickle.dump((graph, bus_stops), f)

# Create environment instance
env = CityTransportEnv.CityTransportEnv(graph, bus_stops, num_buses=2)

# Test environment
state = env.reset()
print("Initial State:", state)

# Run a few steps
for _ in range(5):
    actions = env.action_space.sample()  # Random actions
    state, reward, done, _ = env.step(actions)
    print(f"State: {state}, Reward: {reward}, Done: {done}")
    if done:
        break

env.close()
