import gym_snakegame
import gymnasium as gym

env = gym.make(
    "gym_snakegame/SnakeGame-v0", board_size=10, n_channel=1, n_target=1, render_mode='human'
)

obs, info = env.reset()
for i in range(10000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
env.close()