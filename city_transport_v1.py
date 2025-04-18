from stable_baselines3 import PPO
import pickle
import CityTransportEnv

# Load the graph and bus stops
with open("city_graph_with_bs.pkl", "rb") as f:
    graph, bus_stops = pickle.load(f)

# Example for testing
#print(type(graph))  # Should be a networkx.MultiDiGraph
#print(type(bus_stops))  # Should match the type of your bus stops data


# Create environment
env = CityTransportEnv.CityTransportEnv(graph, bus_stops, num_buses=2)

# Initialize PPO model
model = PPO("MultiInputPolicy", env, verbose=1)

# Train the model
model.learn(total_timesteps=10000)

# Test the trained model
state = env.reset()
for _ in range(10):
    action, _ = model.predict(state)
    state, reward, done, _ = env.step(action)
    env.render()
    if done:
        break
