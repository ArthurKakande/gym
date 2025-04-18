import numpy as np
import random
import gym
from gym import spaces

# Define the Environment with Swarm and RL Coordination
class FleetManagementEnv(gym.Env):
    def __init__(self, num_vehicles=5, num_buses=2, num_customers=5, num_charging_stations=3, grid_size=10):
        super(FleetManagementEnv, self).__init__()
        # Parameters
        self.num_vehicles = num_vehicles
        self.num_buses = num_buses
        self.num_customers = num_customers
        self.num_charging_stations = num_charging_stations
        self.grid_size = grid_size
        self.max_battery = 100
        # Vehicle and bus positions
        self.vehicle_positions = np.random.randint(0, grid_size, size=(num_vehicles, 2))
        self.vehicle_batteries = np.full((num_vehicles,), self.max_battery)
        self.bus_routes = [np.random.randint(0, grid_size, size=(5, 2)) for _ in range(num_buses)]
        self.bus_batteries = np.full((num_buses,), self.max_battery)
        # Charging stations
        self.charging_stations = np.random.randint(0, grid_size, size=(num_charging_stations, 2))
        # Customers
        self.customers = [
            {'pickup': np.random.randint(0, grid_size, size=(1, 2))[0], 'destination': np.random.randint(0, grid_size, size=(1, 2))[0], 'picked_up': False, 'dropped_off': False}
            for _ in range(num_customers)
        ]
        # Action and observation space
        self.action_space = spaces.Discrete(4)  # 0: Up, 1: Down, 2: Left, 3: Right
        self.observation_space = spaces.Box(low=0, high=grid_size, shape=(num_vehicles + num_buses + num_customers, 2), dtype=np.int32)
        # Metrics
        self.total_emissions = 0
        self.total_energy_consumed = 0
        self.total_customer_satisfaction = 0

    def reset(self):
        self.vehicle_positions = np.random.randint(0, self.grid_size, size=(self.num_vehicles, 2))
        self.vehicle_batteries = np.full((self.num_vehicles,), self.max_battery)
        self.bus_positions = [route[0] for route in self.bus_routes]
        self.bus_batteries = np.full((self.num_buses,), self.max_battery)
        self.customers = [
            {'pickup': np.random.randint(0, self.grid_size, size=(1, 2))[0], 'destination': np.random.randint(0, self.grid_size, size=(1, 2))[0], 'picked_up': False, 'dropped_off': False}
            for _ in range(self.num_customers)
        ]
        self.total_emissions = 0
        self.total_energy_consumed = 0
        self.total_customer_satisfaction = 0
        return self._get_state()

    def step(self, actions):
        # Apply RL actions to individual vehicles
        new_vehicle_positions = self.vehicle_positions.copy()
        rewards = [0] * self.num_vehicles

        # Swarm-based global coordination
        swarm_recommendations = self.swarm_controller()  # Swarm determines global goals

        for vehicle_id, action in enumerate(actions):
            pos = self.vehicle_positions[vehicle_id]
            if action == 0:  # Up
                new_vehicle_positions[vehicle_id] = [pos[0], max(0, pos[1] - 1)]
            elif action == 1:  # Down
                new_vehicle_positions[vehicle_id] = [pos[0], min(self.grid_size - 1, pos[1] + 1)]
            elif action == 2:  # Left
                new_vehicle_positions[vehicle_id] = [max(0, pos[0] - 1), pos[1]]
            elif action == 3:  # Right
                new_vehicle_positions[vehicle_id] = [min(self.grid_size - 1, pos[0] + 1), pos[1]]

            # Ensure RL agents adhere to swarm constraints
            if np.linalg.norm(new_vehicle_positions[vehicle_id] - swarm_recommendations[vehicle_id]) > 1:
                rewards[vehicle_id] -= 1  # Penalize deviations from swarm goals

        # Update environment metrics
        self.vehicle_positions = new_vehicle_positions
        self.total_energy_consumed += sum(np.linalg.norm(self.vehicle_positions - swarm_recommendations, axis=1))
        self.total_emissions += self.total_energy_consumed * 0.05
        self._update_customer_status()

        # Check stopping condition
        done = all(battery <= 0 for battery in self.vehicle_batteries)  # Fixed error
        return self._get_state(), rewards, done, {
            'total_emissions': self.total_emissions,
            'total_energy_consumed': self.total_energy_consumed,
            'total_customer_satisfaction': self.total_customer_satisfaction
        }

    def swarm_controller(self):
        # Global optimization logic for the fleet
        recommendations = []
        for i in range(self.num_vehicles):
            closest_customer = min(
                self.customers,
                key=lambda customer: np.linalg.norm(self.vehicle_positions[i] - customer['pickup'])
                if not customer['picked_up'] else float('inf')
            )
            if closest_customer:
                recommendations.append(closest_customer['pickup'])
            else:
                recommendations.append(self.vehicle_positions[i])  # No customer, stay in place
        return np.array(recommendations)

    def _update_customer_status(self):
        for customer in self.customers:
            for vehicle_id, vehicle_pos in enumerate(self.vehicle_positions):
                if not customer['picked_up'] and np.array_equal(vehicle_pos, customer['pickup']):
                    customer['picked_up'] = True
                elif customer['picked_up'] and np.array_equal(vehicle_pos, customer['destination']):
                    customer['dropped_off'] = True
                    self.total_customer_satisfaction += 1

    def _get_state(self):
        return np.concatenate((self.vehicle_positions, [customer['pickup'] for customer in self.customers]), axis=0)

    def render(self):
        print("Vehicle Positions: ", self.vehicle_positions)
        print("Total Customer Satisfaction: ", self.total_customer_satisfaction)


# Define the RL agent
class QLearningAgent:
    def __init__(self, action_space, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}
        self.action_space = action_space

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(range(self.action_space))
        if tuple(state) not in self.q_table:
            self.q_table[tuple(state)] = np.zeros(self.action_space)
        return np.argmax(self.q_table[tuple(state)])

    def update_q_value(self, state, action, reward, next_state):
        if tuple(state) not in self.q_table:
            self.q_table[tuple(state)] = np.zeros(self.action_space)
        if tuple(next_state) not in self.q_table:
            self.q_table[tuple(next_state)] = np.zeros(self.action_space)
        best_next_action = np.argmax(self.q_table[tuple(next_state)])
        self.q_table[tuple(state)][action] += self.alpha * (
            reward + self.gamma * self.q_table[tuple(next_state)][best_next_action] - self.q_table[tuple(state)][action]
        )


# Training loop
def train():
    env = FleetManagementEnv()
    agent = QLearningAgent(action_space=env.action_space.n)
    episodes = 2
    for episode in range(episodes):
        state = env.reset()
        done = False
        while not done:
            actions = [agent.choose_action(state[i]) for i in range(env.num_vehicles)]
            next_state, rewards, done, _ = env.step(actions)
            for i in range(env.num_vehicles):
                agent.update_q_value(state[i], actions[i], rewards[i], next_state[i])
            state = next_state
        print(f"Episode {episode} completed.")

if __name__ == "__main__":
    train()
