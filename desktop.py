import random

import ray
from ray import tune
from ray.rllib.algorithms import PPO, PPOConfig
from ray.tune import sample_from, Trainable
from ray.tune.registry import register_env
from ray.tune.schedulers.pb2 import PB2
import ray.tune.tune
from tune import Trial

from gym_snakegame.envs import SnakeGameEnv


def env_creator():
    return SnakeGameEnv


register_env("snake_game", env_creator())

ray.init(include_dashboard=True)

pb2 = PB2(
    time_attr="timesteps_total",
    perturbation_interval=50000,
    quantile_fraction=0.25,
    hyperparam_bounds={
        "lambda": [0.9, 1.0],
        "clip_param": [0.1, 0.5],
        "lr": [1e-5, 1e-3],
        "train_batch_size": [1000, 60000]
    }
)

analysis = tune.run(
    "PPO",
    name="test",
    resume="True+ERRORED",
    storage_path="/tmp/ray_results",
    scheduler=pb2,
    verbose=1,
    num_samples=5,
    reuse_actors=True,
    config={
        "env": "snake_game",
        "reuse_actors": True,
        "log_level": "INFO",
        "seed": 42,
        "kl_coeff": 1.0,
        "num_gpus": 1,
        "observation_filter": "MeanStdFilter",
        "model": {
            "fcnet_hiddens": [32, 32],
            "free_log_std": True
        },
        "num_sgd_iter": 10,
        "sgd_minibatch_size": 128,
        "lambda": sample_from(lambda spec: random.uniform(0.9, 1.0)),
        "clip_param": sample_from(lambda spec: random.uniform(0.1, 0.5)),
        "lr": sample_from(lambda spec: random.uniform(1e-3, 1e-5)),
        "train_batch_size": sample_from(lambda spec: random.uniform(1000, 60000)),
    },
    metric="episode_reward_mean",
    mode="max"
)

result = analysis.get_best_config()

algo = (PPOConfig()
        .env_runners(num_env_runners=4)
        .resources(num_gpus=0)
        .environment(env="snake_game")
        ).build().from_checkpoint(analysis.get_best_checkpoint(analysis.get_best_trial()))
algo.train()
