import numpy as np
import random
import gym
from gym import spaces

# Define the Environment with Customers
class FleetManagementEnv(gym.Env):
    def __init__(self, num_vehicles=5, num_buses=2, num_customers=5, num_charging_stations=3, grid_size=10):
        super(FleetManagementEnv, self).__init__()
        # Parameters
        self.num_vehicles = num_vehicles
        self.num_buses = num_buses
        self.num_customers = num_customers
        self.num_charging_stations = num_charging_stations
        self.grid_size = grid_size  # Define grid size (rural area)
        self.max_battery = 100  # Max battery capacity for vehicles and buses
        self.max_distance = grid_size * 2  # Max possible distance in the grid
        # Initialize vehicle positions and battery levels
        self.vehicle_positions = np.random.randint(0, grid_size, size=(num_vehicles, 2))
        self.vehicle_batteries = np.full((num_vehicles,), self.max_battery)
        # Initialize bus routes (predefined)
        self.bus_routes = []
        for _ in range(self.num_buses):
            # Random routes with 5 waypoints
            route = np.random.randint(0, grid_size, size=(5, 2))
            self.bus_routes.append(route)
        # Bus battery levels (same as vehicles)
        self.bus_batteries = np.full((num_buses,), self.max_battery)
        # Charging station positions
        self.charging_stations = np.random.randint(0, grid_size, size=(num_charging_stations, 2))
        # Initialize customer positions and destinations
        self.customers = []
        for _ in range(self.num_customers):
            pickup = np.random.randint(0, grid_size, size=(1, 2))[0]
            destination = np.random.randint(0, grid_size, size=(1, 2))[0]
            self.customers.append({'pickup': pickup, 'destination': destination, 'picked_up': False, 'dropped_off': False})
        # Action and observation space
        self.action_space = spaces.Discrete(4)  # 0: Up, 1: Down, 2: Left, 3: Right (for movement)
        self.observation_space = spaces.Box(low=0, high=grid_size, shape=(num_vehicles + num_buses + num_customers, 2), dtype=np.int32)
        # Metrics
        self.total_emissions = 0
        self.total_energy_consumed = 0
        self.total_customer_satisfaction = 0

    def reset(self):
        # Reset vehicle positions, bus routes, customer statuses, and battery levels
        self.vehicle_positions = np.random.randint(0, self.grid_size, size=(self.num_vehicles, 2))
        self.vehicle_batteries = np.full((self.num_vehicles,), self.max_battery)
        self.bus_positions = [route[0] for route in self.bus_routes]  # Starting at first waypoint of each bus
        self.bus_batteries = np.full((self.num_buses,), self.max_battery)
        self.customers = []
        for _ in range(self.num_customers):
            pickup = np.random.randint(0, self.grid_size, size=(1, 2))[0]
            destination = np.random.randint(0, self.grid_size, size=(1, 2))[0]
            self.customers.append({'pickup': pickup, 'destination': destination, 'picked_up': False, 'dropped_off': False})
        self.total_emissions = 0
        self.total_energy_consumed = 0
        self.total_customer_satisfaction = 0
        return np.concatenate((self.vehicle_positions, self.bus_positions, [customer['pickup'] for customer in self.customers]), axis=0)

    def step(self, action):
        # Initialize rewards for all vehicles to 0
        rewards = [0] * self.num_vehicles
        done = False
        new_vehicle_positions = self.vehicle_positions.copy()
        new_bus_positions = self.bus_positions.copy()

        # Move vehicles based on RL action (dynamic movement)
        for vehicle_id in range(self.num_vehicles):
            pos = self.vehicle_positions[vehicle_id]
            if action[vehicle_id] == 0:  # Up
                new_vehicle_positions[vehicle_id] = [pos[0], max(0, pos[1] - 1)]
            elif action[vehicle_id] == 1:  # Down
                new_vehicle_positions[vehicle_id] = [pos[0], min(self.grid_size - 1, pos[1] + 1)]
            elif action[vehicle_id] == 2:  # Left
                new_vehicle_positions[vehicle_id] = [max(0, pos[0] - 1), pos[1]]
            elif action[vehicle_id] == 3:  # Right
                new_vehicle_positions[vehicle_id] = [min(self.grid_size - 1, pos[0] + 1), pos[1]]

            # Update battery (energy consumption)
            distance_traveled = np.linalg.norm(np.array(new_vehicle_positions[vehicle_id]) - np.array(pos))
            energy_used = distance_traveled * 0.1  # Assume 0.1 battery usage per unit distance
            self.vehicle_batteries[vehicle_id] -= energy_used
            self.total_energy_consumed += energy_used

            # Calculate emissions (based on energy used)
            emissions = energy_used * 0.05  # Assume 0.05 carbon emissions per unit energy
            self.total_emissions += emissions

            # Customer pickup/dropoff logic
            for customer in self.customers:
                if not customer['picked_up'] and np.array_equal(new_vehicle_positions[vehicle_id], customer['pickup']):
                    customer['picked_up'] = True
                    rewards[vehicle_id] += 5  # Reward for picking up customer
                elif customer['picked_up'] and not customer['dropped_off'] and np.array_equal(new_vehicle_positions[vehicle_id], customer['destination']):
                    customer['dropped_off'] = True
                    rewards[vehicle_id] += 10  # Reward for dropping off customer
                    self.total_customer_satisfaction += 1  # Satisfaction for successful trip

        # Move buses along their predefined route (sequentially move to next waypoint)
        for bus_id in range(self.num_buses):
            current_position = new_bus_positions[bus_id]
            route = self.bus_routes[bus_id]
            # Find the next waypoint in the route
            current_index = np.where(np.all(route == current_position, axis=1))[0][0]
            next_index = (current_index + 1) % len(route)  # Wrap around to loop the route
            new_bus_positions[bus_id] = route[next_index]

            # Update battery (energy consumption)
            distance_traveled = np.linalg.norm(np.array(new_bus_positions[bus_id]) - np.array(current_position))
            energy_used = distance_traveled * 0.1  # Assume 0.1 battery usage per unit distance
            self.bus_batteries[bus_id] -= energy_used
            self.total_energy_consumed += energy_used

            # Calculate emissions (based on energy used)
            emissions = energy_used * 0.05  # Assume 0.05 carbon emissions per unit energy
            self.total_emissions += emissions

        # Check if all vehicles and buses are out of energy or a stopping condition is met
        if np.all(self.vehicle_batteries <= 0) and np.all(self.bus_batteries <= 0):
            done = True

        # Return new state and metrics
        state = np.concatenate((new_vehicle_positions, new_bus_positions, [customer['pickup'] for customer in self.customers]), axis=0)
        return state, rewards, done, {
            'total_emissions': self.total_emissions,
            'total_energy_consumed': self.total_energy_consumed,
            'total_customer_satisfaction': self.total_customer_satisfaction
        }

    def render(self):
        print("Vehicle Positions: ", self.vehicle_positions)
        print("Bus Positions: ", self.bus_positions)
        print("Customer Pickups: ", [customer['pickup'] for customer in self.customers])
        print("Customer Destinations: ", [customer['destination'] for customer in self.customers])
        print("Customer Satisfaction: ", self.total_customer_satisfaction)
        print("Total Emissions: ", self.total_emissions)
        print("Total Energy Consumed: ", self.total_energy_consumed)


# Define Q-learning agent for vehicles
class QLearningAgent:
    def __init__(self, action_space, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.q_table = {}  # Q-table for storing Q-values for state-action pairs
        self.action_space = action_space

    def choose_action(self, state):
        # Epsilon-greedy policy
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(range(self.action_space))
        else:
            if tuple(state) not in self.q_table:
                self.q_table[tuple(state)] = np.zeros(self.action_space)
            return np.argmax(self.q_table[tuple(state)])

    def update_q_value(self, state, action, reward, next_state):
        if tuple(state) not in self.q_table:
            self.q_table[tuple(state)] = np.zeros(self.action_space)
        if tuple(next_state) not in self.q_table:
            self.q_table[tuple(next_state)] = np.zeros(self.action_space)
        # Q-learning update rule
        best_next_action = np.argmax(self.q_table[tuple(next_state)])
        old_q_value = self.q_table[tuple(state)][action]
        new_q_value = old_q_value + self.alpha * (reward + self.gamma * self.q_table[tuple(next_state)][best_next_action] - old_q_value)
        self.q_table[tuple(state)][action] = new_q_value


# Training loop
def train():
    num_vehicles = 5
    num_buses = 2
    num_customers = 5
    env = FleetManagementEnv(num_vehicles=num_vehicles, num_buses=num_buses, num_customers=num_customers)
    agent = QLearningAgent(action_space=env.action_space.n)
    episodes = 1000
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            # Choose actions for each vehicle
            actions = [agent.choose_action(state[vehicle_id]) for vehicle_id in range(num_vehicles)]
            next_state, rewards, done, metrics = env.step(actions)
            # Update Q-values for each vehicle
            for vehicle_id in range(num_vehicles):
                agent.update_q_value(state[vehicle_id], actions[vehicle_id], rewards[vehicle_id], next_state[vehicle_id])
            state = next_state
            total_reward += sum(rewards)
        # Output metrics for each episode
        print(f"Episode {episode}, Total Reward: {total_reward}")
        print(f"Emissions: {metrics['total_emissions']}, Energy Consumed: {metrics['total_energy_consumed']}, Customer Satisfaction: {metrics['total_customer_satisfaction']}")
    return agent, env


if __name__ == "__main__":
    agent, env = train()
    env.render()  # Show final state and metrics
