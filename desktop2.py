import os
from ray import tune
from ray.rllib.algorithms import PPOConfig
from ray.tune import register_env
from gym_snakegame.envs import SnakeGameEnv


def env_creator():
    return SnakeGameEnv


register_env("snake_game", env_creator())

algo = (PPOConfig()
        .training(lr=0.0010000000474974513,
                  gamma=0.99,
                  clip_param=0.11818516254425049,
                  lambda_=1.0,
                  train_batch_size=1000)
        .env_runners(num_env_runners=8)
        .resources(num_gpus=0)
        .environment(env="snake_game",
                     render_env=True)
        ).build().from_checkpoint("/tmp/ray_results/snake")

episodes = 10000
i = 0
while True:
    print("Episode: " + str(i+1))
    algo.train()
    if (i+1) % 10 == 0:
        algo.save("/tmp/ray_results/snake")
    i += 1

game = SnakeGameEnv(render_mode='human')
obs, info = game.reset()

action = algo.compute_single_action(observation=obs, info=info)
obs, prev_reward, terminated, x, info = game.step(action)

while True:
    action = algo.compute_single_action(observation=obs, info=info, prev_action=game.prev_action, prev_reward=prev_reward)
    obs, prev_reward, terminated, x, info = game.step(action)
    if terminated:
        game.reset()
