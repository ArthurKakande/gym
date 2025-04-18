import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import gym
from gym import spaces


class FleetManagementEnv(gym.Env):
    def __init__(self, num_vehicles=5, num_buses=2, num_customers=50, num_charging_stations=3, grid_size=10, max_steps=1000):
        super(FleetManagementEnv, self).__init__()
        self.num_vehicles = num_vehicles
        self.num_buses = num_buses
        self.num_customers = num_customers
        self.num_charging_stations = num_charging_stations
        self.grid_size = grid_size
        self.max_battery = 50
        self.step_count = 0
        self.max_steps = max_steps

        # Positions and states
        self.vehicle_positions = np.random.randint(0, grid_size, size=(num_vehicles, 2))
        self.vehicle_batteries = np.full((num_vehicles,), self.max_battery)
        self.bus_routes = [np.random.randint(0, grid_size, size=(5, 2)) for _ in range(num_buses)]
        self.bus_positions = [route[0] for route in self.bus_routes]
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

        # Observation and action spaces
        self.observation_space = spaces.Box(low=0, high=grid_size, shape=(num_vehicles + num_buses + num_customers, 2), dtype=np.int32)
        self.action_space = spaces.Discrete(4)  # Actions: 0: Up, 1: Down, 2: Left, 3: Right

        # Metrics
        self.total_emissions = 0
        self.total_energy_consumed = 0
        self.total_customer_satisfaction = 0

    def reset(self):
        self.vehicle_positions = np.random.randint(0, self.grid_size, size=(self.num_vehicles, 2))
        self.vehicle_batteries = np.full((self.num_vehicles,), self.max_battery)
        self.bus_positions = [route[0] for route in self.bus_routes]
        self.bus_batteries = np.full((self.num_buses,), self.max_battery)
        self.step_count = 0
        self.total_emissions = 0
        self.total_energy_consumed = 0
        self.total_customer_satisfaction = 0
        return self._get_state()

    def step(self, actions):
        self.step_count += 1

        # Move vehicles based on actions
        for vehicle_id, action in enumerate(actions[:self.num_vehicles]):
            pos = self.vehicle_positions[vehicle_id]
            if action == 0:  # Up
                self.vehicle_positions[vehicle_id][1] = max(0, pos[1] - 1)
            elif action == 1:  # Down
                self.vehicle_positions[vehicle_id][1] = min(self.grid_size - 1, pos[1] + 1)
            elif action == 2:  # Left
                self.vehicle_positions[vehicle_id][0] = max(0, pos[0] - 1)
            elif action == 3:  # Right
                self.vehicle_positions[vehicle_id][0] = min(self.grid_size - 1, pos[0] + 1)

            # Battery consumption
            self.vehicle_batteries[vehicle_id] -= 1  # Simplified consumption model

        # Update bus positions along predefined routes
        for bus_id in range(self.num_buses):
            route = self.bus_routes[bus_id]
            route_idx = (self.step_count // 5) % len(route)  # Cycle through the route
            self.bus_positions[bus_id] = route[route_idx]

        # Update customer status
        self._update_customer_status()

        # Calculate emissions and energy consumption
        self.total_energy_consumed += np.sum([np.linalg.norm(self.vehicle_positions[i] - self.bus_positions[i % self.num_buses]) for i in range(self.num_vehicles)])
        self.total_emissions = self.total_energy_consumed * 0.05

        # Check termination
        done = self.step_count >= self.max_steps or all(battery <= 0 for battery in self.vehicle_batteries)
        rewards = [1 if customer['dropped_off'] else -1 for customer in self.customers]
        return self._get_state(), rewards, done, {
            'total_emissions': self.total_emissions,
            'total_energy_consumed': self.total_energy_consumed,
            'total_customer_satisfaction': self.total_customer_satisfaction
        }

    def _update_customer_status(self):
        for customer in self.customers:
            for vehicle_pos in self.vehicle_positions:
                if not customer['picked_up'] and np.array_equal(vehicle_pos, customer['pickup']):
                    customer['picked_up'] = True
                if customer['picked_up'] and np.array_equal(vehicle_pos, customer['destination']):
                    customer['dropped_off'] = True
                    self.total_customer_satisfaction += 1

    def _get_state(self):
        return np.concatenate((
            self.vehicle_positions, 
            self.bus_positions, 
            [customer['pickup'] for customer in self.customers]
        ))

    def render(self):
        print(f"Vehicles: {self.vehicle_positions}")
        print(f"Buses: {self.bus_positions}")
        print(f"Customer Satisfaction: {self.total_customer_satisfaction}")


# Adjust QLearningAgent and QMixer as required by the new environment
class QLearningAgent:
    def __init__(self, state_size, action_space, learning_rate=0.01, discount_factor=0.99, exploration_rate=1.0, exploration_decay=0.995):
        self.state_size = state_size
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay

        # Neural network to approximate Q-values
        self.model = nn.Sequential(
            nn.Linear(self.state_size, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_space)
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def choose_action(self, state):
        # Epsilon-greedy policy
        if random.uniform(0, 1) < self.exploration_rate:
            return random.choice(range(self.action_space))
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
            q_values = self.model(state_tensor)
            return torch.argmax(q_values).item()

    def update_q_values(self, state, action, reward, next_state, done):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
        reward_tensor = torch.tensor([reward], dtype=torch.float32)

        # Compute target
        with torch.no_grad():
            target_q = reward_tensor
            if not done:
                next_q_values = self.model(next_state_tensor)
                target_q += self.discount_factor * torch.max(next_q_values)

        # Compute predicted Q-value
        predicted_q = self.model(state_tensor)[0, action]

        # Compute loss and update network
        loss = self.criterion(predicted_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def decay_exploration(self):
        self.exploration_rate *= self.exploration_decay

class QMixer:
    def __init__(self, num_agents, state_size, mixing_size=32):
        self.num_agents = num_agents

        # Mixing network
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_size, mixing_size),
            nn.ReLU(),
            nn.Linear(mixing_size, num_agents * mixing_size)
        )
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_size, mixing_size),
            nn.ReLU(),
            nn.Linear(mixing_size, mixing_size)
        )
        self.hyper_b = nn.Linear(state_size, mixing_size)

    def forward(self, agent_qs, state):
        # Reshape agent Qs and state
        agent_qs = agent_qs.view(-1, self.num_agents, 1)
        state_tensor = torch.tensor(state, dtype=torch.float32)

        # Hypernetwork outputs
        w1 = self.hyper_w1(state_tensor).view(-1, self.num_agents, mixing_size)
        b1 = self.hyper_b(state_tensor).view(-1, 1, mixing_size)
        w2 = self.hyper_w2(state_tensor).view(-1, mixing_size, 1)

        # Forward pass
        hidden = torch.relu(torch.bmm(agent_qs, w1) + b1)
        q_total = torch.bmm(hidden, w2)
        return q_total

# Update the training loop to integrate metrics printing
def train():
    # Initialize environment and agents
    env = FleetManagementEnv()
    num_agents = env.num_vehicles
    state_size = (env.num_vehicles + env.num_buses + env.num_customers) * 2
    action_space = env.action_space.n

    # Initialize agents
    agents = [QLearningAgent(state_size=state_size, action_space=action_space) for _ in range(num_agents)]
    episodes = 2
    metrics = []

    for episode in range(episodes):
        print(f"Starting Episode {episode + 1}...")
        state = env.reset()
        done = False
        episode_reward = 0
        total_emissions = 0
        total_energy_consumed = 0

        while not done:
            actions = [agent.choose_action(state.flatten()) for agent in agents]
            next_state, rewards, done, info = env.step(actions)

            for i, agent in enumerate(agents):
                agent.update_q_values(
                    state=state.flatten(), action=actions[i], reward=rewards[i],
                    next_state=next_state.flatten(), done=done
                )

            state = next_state
            episode_reward += sum(rewards)
            total_emissions = info['total_emissions']
            total_energy_consumed = info['total_energy_consumed']

            # Debug print for each step
            print(f"Step {env.step_count}: Reward = {sum(rewards)}, Emissions = {total_emissions:.2f}, Energy = {total_energy_consumed:.2f}")

        for agent in agents:
            agent.decay_exploration()

        print(f"Episode {episode + 1} completed.")
        print(f"  Total Reward: {episode_reward}")
        print(f"  Total Customer Satisfaction: {info['total_customer_satisfaction']}")
        print(f"  Total Carbon Emissions: {total_emissions:.2f}")
        print(f"  Total Energy Consumed: {total_energy_consumed:.2f}")

        metrics.append({
            'reward': episode_reward,
            'emissions': total_emissions,
            'energy_consumed': total_energy_consumed,
            'customer_satisfaction': info['total_customer_satisfaction']
        })

    print("Training complete.")
    return metrics

if __name__ == "__main__":
    train()