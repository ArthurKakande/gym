import gym
from gym import spaces
import numpy as np
import networkx as nx

class CityTransportEnv(gym.Env):
    """
    Custom RL environment for public transport simulation.
    """
    def __init__(self, graph, bus_stops, num_buses=1):
        super(CityTransportEnv, self).__init__()
        
        # Save graph and bus stop data
        self.graph = graph
        self.bus_stops = bus_stops

        # Define action and observation spaces
        self.num_buses = num_buses
        self.nodes = list(graph.nodes)

        # Action space: Move each bus to a neighboring node
        self.action_space = spaces.MultiDiscrete([len(self.nodes)] * num_buses)

        # Observation space: Current position of buses and traffic states
        self.observation_space = spaces.Dict({
            "bus_positions": spaces.MultiDiscrete([len(self.nodes)] * num_buses),
            "traffic_states": spaces.Box(low=0, high=1, shape=(len(self.nodes),), dtype=np.float32),
        })

        # Initialize state
        self.state = {
            "bus_positions": np.random.choice(self.nodes, size=num_buses),
            "traffic_states": np.zeros(len(self.nodes)),
        }

        # Rewards
        self.total_travel_time = 0

    def step(self, actions):
        """
        Perform one step in the environment.
        """
        rewards = []
        for i, action in enumerate(actions):
            current_position = self.state["bus_positions"][i]
            neighbors = list(self.graph.neighbors(current_position))
            
            if action < len(neighbors):
                # Move the bus to the chosen neighbor
                next_position = neighbors[action]
                self.state["bus_positions"][i] = next_position

                # Calculate travel time
                travel_time = self.graph[current_position][next_position][0]["travel_time"]
                self.total_travel_time += travel_time

                # Reward: Negative travel time (minimize travel time)
                rewards.append(-travel_time)
            else:
                # Invalid action, apply a penalty
                rewards.append(-10)

        # Update traffic states (can be dynamic in advanced versions)
        self.state["traffic_states"] = np.random.random(len(self.nodes))

        # Compute the overall reward
        reward = sum(rewards)

        # Check if episode is done
        done = self.total_travel_time > 1000  # Arbitrary condition

        return self.state, reward, done, {}

    def reset(self):
        """
        Reset the environment to the initial state.
        """
        self.state = {
            "bus_positions": np.random.choice(self.nodes, size=self.num_buses),
            "traffic_states": np.zeros(len(self.nodes)),
        }
        self.total_travel_time = 0
        return self.state

    def render(self, mode="human"):
        """
        Visualize the environment.
        """
        print(f"Bus positions: {self.state['bus_positions']}")
        print(f"Traffic states: {self.state['traffic_states'][:10]}... (truncated)")

    def close(self):
        """
        Close the environment.
        """
        pass
