import gymnasium as gym

def run():
    # Create the Frozen Lake environment
    env = gym.make("FrozenLake-v1", is_slippery=True, map_name="8x8", render_mode="human")  

    # Reset the environment to the initial state
    state = env.reset()[0]
    terminated = False
    truncated = False

    # Run the environment until the episode is done
    while (not terminated and not truncated):
        # Render the current state of the environment
        env.render()

        # Take a random action
        action = env.action_space.sample()

        # Apply the action to the environment
        new_state, reward, terminated, truncated, info = env.step(action)

        state = new_state

    # Close the environment
    env.close()

if __name__ == "__main__":
    run()