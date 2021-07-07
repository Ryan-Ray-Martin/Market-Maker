"""
Adopted from Alexey Bakshaev's "Market-making with reinforcement-learning (SAC)" , 2000
Created by Ryan Martin, 2021
"""
import matplotlib.pyplot as plt
from statistics import mean
import dashboard as db

from hedging_env import HedgingEnv, RewardType
from custom_fcnn import TorchCustomModel
import os

import ray
from ray import tune
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.registry import register_env

torch, nn = try_import_torch()

def env_creator(env_name):
    if env_name == "HedgeEnv-v0":
        from hedging_env import HedgingEnv as env
    else:
        raise NotImplementedError
    return env

env = env_creator("HedgeEnv-v0")
tune.register_env('myEnv', lambda config: env(use_skew=False, reward_type=RewardType.MaxPnl))
    
plt.ion()

def callback(locals, globals):
    env.hedger.verbose = False
    reward = 0
    pos = locals['self'].replay_buffer.pos
    if pos > 0: reward = locals['self'].replay_buffer.rewards[pos-1]

    global reward_history
    reward_history += [reward]
    if env.timestamp % 5 == 0:
        db.plot_rewards(env, reward_history)

env = HedgingEnv(use_skew=False, reward_type=RewardType.MaxPnl)
env.model_name = "sac_autohedger_single"

policy_args = {"net_arch":  [1024, 1024]}

reward_history = []
noise = NormalActionNoise(0, 50)
model = SAC(MlpPolicy, env,
            verbose=2,
            learning_rate=7e-6,
            target_update_interval=32,
            learning_starts=20,
            use_sde_at_warmup=True,
            use_sde=False,
            policy_kwargs=policy_args,
            )

model.learn(total_timesteps=6000, log_interval=50, n_eval_episodes=1000, callback=callback)
# to save the model uncomment
#model.save(os.path.dirname(__file__) + "/models/" + env.model_name)
#model.save("./models/" + env.model_name)
plt.ioff()

del model