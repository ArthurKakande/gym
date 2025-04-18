import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import random

# Simulated environment class
class DeliveryEnvironment:
    def __init__(self, size=10):
        self.size = size
        self.grid = np.zeros((size, size))
        self.robot_pos = [0, 0]
        self.delivery_point = [size-1, size-1]
        self.obstacles = self._generate_obstacles()

    def _generate_obstacles(self):
        obstacles = []
        for _ in range(self.size * 2):
            pos = [random.randint(0, self.size-1), random.randint(0, self.size-1)]
            if pos != self.robot_pos and pos != self.delivery_point:
                obstacles.append(pos)
        return obstacles

    def get_state(self):
        # Create state vector: [robot_x, robot_y, delivery_x, delivery_y, 
        #                      distance_to_nearest_obstacle]
        state = np.array(self.robot_pos + self.delivery_point)
        min_obstacle_dist = float('inf')
        for obs in self.obstacles:
            dist = np.sqrt((self.robot_pos[0]-obs[0])**2 + 
                          (self.robot_pos[1]-obs[1])**2)
            min_obstacle_dist = min(min_obstacle_dist, dist)
        return np.append(state, min_obstacle_dist)

    def step(self, action):
        # Actions: 0=up, 1=right, 2=down, 3=left
        moves = [(-1,0), (0,1), (1,0), (0,-1)]
        new_pos = [self.robot_pos[0] + moves[action][0],
                  self.robot_pos[1] + moves[action][1]]
        
        # Check boundaries
        if 0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size:
            # Check obstacles
            if new_pos not in self.obstacles:
                self.robot_pos = new_pos

        # Calculate reward
        if self.robot_pos == self.delivery_point:
            reward = 100  # Delivery completed
            done = True
        elif self.robot_pos in self.obstacles:
            reward = -50  # Hit obstacle
            done = True
        else:
            reward = -1  # Time penalty
            done = False

        return self.get_state(), reward, done

# Neural Network Model
def create_model(state_size=5, action_size=4):
    model = keras.Sequential([
        layers.Dense(24, activation='relu', input_shape=(state_size,)),
        layers.Dense(24, activation='relu'),
        layers.Dense(action_size, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Training function
def train_delivery_robot(episodes=1000):
    env = DeliveryEnvironment()
    state_size = 5
    action_size = 4
    model = create_model(state_size, action_size)
    
    # Training parameters
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995
    gamma = 0.95
    batch_size = 32
    memory = []

    for episode in range(episodes):
        state = env.get_state()
        total_reward = 0
        
        for time in range(200):  # Max 200 steps per episode
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = random.randint(0, 3)
            else:
                action = np.argmax(model.predict(
                    state.reshape(1, -1), verbose=0)[0])

            # Take action and observe result
            next_state, reward, done = env.step(action)
            total_reward += reward

            # Store experience
            memory.append((state, action, reward, next_state, done))
            if len(memory) > 2000:
                memory.pop(0)

            # Training
            if len(memory) >= batch_size:
                batch = random.sample(memory, batch_size)
                states = np.array([x[0] for x in batch])
                next_states = np.array([x[3] for x in batch])
                
                # Compute Q-values
                current_q = model.predict(states, verbose=0)
                next_q = model.predict(next_states, verbose=0)
                
                # Update Q-values
                for i, (_, action, reward, _, done) in enumerate(batch):
                    if done:
                        current_q[i][action] = reward
                    else:
                        current_q[i][action] = reward + gamma * np.max(next_q[i])
                
                # Train model
                model.fit(states, current_q, epochs=1, verbose=0)

            state = next_state
            if done:
                break

        # Decay epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
        if episode % 100 == 0:
            print(f"Episode: {episode}, Total Reward: {total_reward}")

    return model

# Example usage
if __name__ == "__main__":
    trained_model = train_delivery_robot(1000) 