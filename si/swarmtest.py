import numpy as np
import random
import gym
from gym import spaces

class FleetManagementEnv(gym.Env):
    def __init__(self, num_vehicles=5, num_buses=2, num_customers=100, num_charging_stations=3, grid_size=10, max_steps=10000):
        super(FleetManagementEnv, self).__init__()
        self.num_vehicles = num_vehicles
        self.num_buses = num_buses
        self.num_customers = num_customers
        self.num_charging_stations = num_charging_stations
        self.grid_size = grid_size
        self.max_battery = 50
        self.step_count = 0
        self.max_steps = max_steps

        # Vehicle positions and batteries
        self.vehicle_positions = np.random.randint(0, grid_size, size=(num_vehicles, 2))
        self.vehicle_batteries = np.full((num_vehicles,), self.max_battery)
        self.bus_routes = [np.random.randint(0, grid_size, size=(5, 2)) for _ in range(num_buses)]
        self.bus_batteries = np.full((num_buses,), self.max_battery)

        # Charging stations
        self.charging_stations = np.random.randint(0, grid_size, size=(num_charging_stations, 2))

        # Customers
        self.customers = [
            {'pickup': np.random.randint(0, grid_size, size=(1, 2))[0], 
             'destination': np.random.randint(0, grid_size, size=(1, 2))[0], 
             'picked_up': False, 'dropped_off': False}
            for _ in range(num_customers)
        ]

        # Action and observation space
        self.action_space = spaces.Discrete(4)  # 0: Up, 1: Down, 2: Left, 3: Right
        self.observation_space = spaces.Box(low=0, high=grid_size, shape=(num_vehicles + num_customers, 2), dtype=np.int32)

        # Metrics
        self.total_emissions = 0
        self.total_energy_consumed = 0
        self.total_customer_satisfaction = 0

    def reset(self):
        self.vehicle_positions = np.random.randint(0, self.grid_size, size=(self.num_vehicles, 2))
        self.vehicle_batteries = np.full((self.num_vehicles,), self.max_battery)
        self.customers = [
            {'pickup': np.random.randint(0, self.grid_size, size=(1, 2))[0], 
             'destination': np.random.randint(0, self.grid_size, size=(1, 2))[0], 
             'picked_up': False, 'dropped_off': False}
            for _ in range(self.num_customers)
        ]
        self.total_emissions = 0
        self.total_energy_consumed = 0
        self.total_customer_satisfaction = 0
        self.step_count = 0
        return self._get_state()

    def step(self, actions):
        self.step_count += 1

        new_vehicle_positions = self.vehicle_positions.copy()
        rewards = [0] * self.num_vehicles

        # Swarm-based global coordination
        swarm_recommendations = self.swarm_controller()

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

            # Battery consumption
            distance_moved = np.linalg.norm(new_vehicle_positions[vehicle_id] - pos)
            self.vehicle_batteries[vehicle_id] -= distance_moved * 5  # Adjust energy cost

            # Reward for moving towards swarm recommendations
            rewards[vehicle_id] += -np.linalg.norm(new_vehicle_positions[vehicle_id] - swarm_recommendations[vehicle_id])

        self.vehicle_positions = new_vehicle_positions
        self._update_customer_status()

        # Update metrics
        self.total_energy_consumed += np.sum([np.linalg.norm(pos - rec) for pos, rec in zip(self.vehicle_positions, swarm_recommendations)])
        self.total_emissions += self.total_energy_consumed * 0.05

        # Stopping condition
        done = all(battery <= 0 for battery in self.vehicle_batteries) or self.step_count >= self.max_steps

        return self._get_state(), rewards, done, {
            'total_emissions': self.total_emissions,
            'total_energy_consumed': self.total_energy_consumed,
            'total_customer_satisfaction': self.total_customer_satisfaction
        }

    def swarm_controller(self):
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
                recommendations.append(self.vehicle_positions[i])
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
        print("Battery Levels: ", self.vehicle_batteries)
        print("Customer Satisfaction: ", self.total_customer_satisfaction)

class QLearningAgent:
    def __init__(self, action_space, learning_rate=0.1, discount_factor=0.99, exploration_rate=1.0, exploration_decay=0.995):
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.q_table = {}  # Dictionary to store state-action values

    def choose_action(self, state):
        # Convert state to a tuple (hashable type)
        state_key = tuple(state.flatten())
        if random.uniform(0, 1) < self.exploration_rate:
            # Explore: choose a random action
            return random.choice(range(self.action_space))
        else:
            # Exploit: choose the action with the highest Q-value
            return self.get_best_action(state_key)

    def get_best_action(self, state_key):
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_space)
        return np.argmax(self.q_table[state_key])

    def update_q_value(self, state, action, reward, next_state):
        state_key = tuple(state.flatten())
        next_state_key = tuple(next_state.flatten())

        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_space)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_space)

        # Q-learning update rule
        best_next_action = self.get_best_action(next_state_key)
        td_target = reward + self.discount_factor * self.q_table[next_state_key][best_next_action]
        td_error = td_target - self.q_table[state_key][action]
        self.q_table[state_key][action] += self.learning_rate * td_error

    def decay_exploration(self):
        self.exploration_rate *= self.exploration_decay


# Training loop
def train():
    env = FleetManagementEnv()
    agent = QLearningAgent(action_space=env.action_space.n)
    episodes = 1000
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        total_emissions = 0
        total_energy_consumed = 0
        
        while not done:
            actions = [agent.choose_action(state[i]) for i in range(env.num_vehicles)]
            next_state, rewards, done, info = env.step(actions)
            
            for i in range(env.num_vehicles):
                agent.update_q_value(state[i], actions[i], rewards[i], next_state[i])
            
            state = next_state
            total_reward += sum(rewards)
            total_emissions = info['total_emissions']  # Retrieve emissions from info
            total_energy_consumed = info['total_energy_consumed']  # Retrieve energy consumption from info

        agent.decay_exploration()
        print(f"Episode {episode + 1} completed.")
        print(f"  Total Reward: {total_reward}")
        print(f"  Total Customer Satisfaction: {info['total_customer_satisfaction']}")
        print(f"  Total Carbon Emissions: {total_emissions:.2f}")
        print(f"  Total Energy Consumed: {total_energy_consumed:.2f}")
    print("Training complete.")

if __name__ == "__main__":
    train()
