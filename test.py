import gymnasium as gym

if __name__ == '__main__':
    env = gym.make('InvertedDoublePendulum-v4', render_mode='rgb_array')
    obs, info = env.reset()

    for i in range(1020):
        action = env.action_space.sample()
        print(f"Action {i}: {action}")
        rgb_array = env.render()
        print(f"Rendered image shape: {rgb_array.shape}")
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i}: {obs}, {reward}, {terminated}, {truncated}, {info}")
        if terminated:
            break

    env.close()