"""
Adopted from Alexey Bakshaev's "Market-making with reinforcement-learning (SAC)" , 2000
Created by Ryan Martin, 2021
"""
import matplotlib.pyplot as plt
from statistics import mean
import dashboard as db

from hedging_env import HedgingEnv, RewardType
import os

import ray
from ray import tune
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.utils.framework import try_import_tf

tf1, tf, tfv = try_import_tf()

def env_creator(env_name):
    if env_name == "HedgeEnv-v0":
        from hedging_env import HedgingEnv as env
    else:
        raise NotImplementedError
    return env
        
env = env_creator("HedgeEnv-v0")
tune.register_env('myEnv', lambda config: env(use_skew=False, reward_type=RewardType.MaxPnl))

class CustomModel(TFModelV2):
    """Example of a keras custom model that just delegates to an fc-net."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        super(CustomModel, self).__init__(obs_space, action_space, num_outputs,
                                          model_config, name)
        self.model = FullyConnectedNetwork(obs_space, action_space,
                                           num_outputs, model_config, name)

    def forward(self, input_dict, state, seq_lens):
        return self.model.forward(input_dict, state, seq_lens)

    def value_function(self):
        return self.model.value_function()

class Training(object):
    def __init__(self):
        ray.shutdown()
        ray.init(num_cpus=16, num_gpus=0, ignore_reinit_error=True)
        #ModelCatalog.register_custom_model("my_model", CustomModel)
        #ModelCatalog.register_custom_model("attention_net", GTrXLNet)
        self.run = "PPO"
        self.config_model = {
            "env": "myEnv",  
            # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
            "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
            "model": {
                #"custom_model": "my_model",
                #"conv_filters": None,
                #"conv_activation": "relu",
                #"use_lstm": True,
                #"lstm_use_prev_action": True,
                #"lstm_use_prev_reward": True,
                "vf_share_layers": False,
            },
            "batch_mode": "truncate_episodes",
            "sgd_minibatch_size": 32,
            "num_sgd_iter": 10,
            "lr": 3e-3,  # try different lrs
            "num_workers": 1,  # parallelism
            "framework": "tf",
        }
        self.config_model["num_workers"] = 0
        self.config_model["exploration_config"] = {
            "type": "Curiosity",  # <- Use the Curiosity module for exploring.
            "eta": 1.0,  # Weight for intrinsic rewards before being added to extrinsic ones.
            "lr": .001,  # Learning rate of the curiosity (ICM) module.
            "feature_dim": 288,  # Dimensionality of the generated feature vectors.
            # Setup of the feature net (used to encode observations into feature (latent) vectors).
            "feature_net_config": {
                "fcnet_hiddens": [],
                "fcnet_activation": "relu",
            },
            "inverse_net_hiddens": [256],  # Hidden layers of the "inverse" model.
            "inverse_net_activation": "relu",  # Activation of the "inverse" model.
            "forward_net_hiddens": [256],  # Hidden layers of the "forward" model.
            "forward_net_activation": "relu",  # Activation of the "forward" model.
            "beta": 0.2,  # Weight for the "forward" loss (beta) over the "inverse" loss (1.0 - beta).
            # Specify, which exploration sub-type to use (usually, the algo's "default"
            # exploration, e.g. EpsilonGreedy for DQN, StochasticSampling for PG/SAC).
            "sub_exploration": {
                "type": "StochasticSampling",
            }
        }

        self.stop = {
            "training_iteration": 100,
            "timesteps_total": 100000,
            "episode_reward_mean": 0.25,
        }

    def train(self):
        """
        Train an RLlib PPO agent using tune until any of the configured stopping criteria is met.
        :param stop_criteria: Dict with stopping criteria.
        See https://docs.ray.io/en/latest/tune/api_docs/execution.html#tune-run
        :return: Return the path to the saved agent (checkpoint) and tune's ExperimentAnalysis object
        See https://docs.ray.io/en/latest/tune/api_docs/analysis.html#experimentanalysis-tune-experimentanalysis
        """
        # make directory for saves
        # init directory in which to save checkpoints
        saves_root = "saves"
        shutil.rmtree(saves_root, ignore_errors=True, onerror=None)

        # init directory in which to log results
        ray_results = "{}/ray_results/".format(os.getenv("HOME"))
        shutil.rmtree(ray_results, ignore_errors=True, onerror=None)

        analysis = ray.tune.run(
            ray.rllib.agents.ppo.PPOTrainer,
            config=self.config_model,
            local_dir=saves_root,
            stop=self.stop,
            checkpoint_at_end = True)

        # list of lists: one list per checkpoint; each checkpoint list contains 1st the path, 2nd the metric value
        checkpoints = analysis.get_trial_checkpoints_paths(
            trial=analysis.get_best_trial(
                'episode_reward_mean',
                mode="max",
                scope="all",
                filter_nan_and_inf=True),
                metric='episode_reward_mean')
        # retrieve the checkpoint path; we only have a single checkpoint, so take the first one
        checkpoint_path = checkpoints[0][0]
        
        return checkpoint_path, analysis
    def train(self):
        """
        Train an RLlib PPO agent using tune until any of the configured stopping criteria is met.
        :param stop_criteria: Dict with stopping criteria.
        See https://docs.ray.io/en/latest/tune/api_docs/execution.html#tune-run
        :return: Return the path to the saved agent (checkpoint) and tune's ExperimentAnalysis object
        See https://docs.ray.io/en/latest/tune/api_docs/analysis.html#experimentanalysis-tune-experimentanalysis
        """
        # make directory for saves
        # init directory in which to save checkpoints
        saves_root = "saves"
        shutil.rmtree(saves_root, ignore_errors=True, onerror=None)

        # init directory in which to log results
        ray_results = "{}/ray_results/".format(os.getenv("HOME"))
        shutil.rmtree(ray_results, ignore_errors=True, onerror=None)

        analysis = ray.tune.run(
            ray.rllib.agents.ppo.PPOTrainer,
            config=self.config_model,
            local_dir=saves_root,
            stop=self.stop,
            checkpoint_at_end = True)

        # list of lists: one list per checkpoint; each checkpoint list contains 1st the path, 2nd the metric value
        checkpoints = analysis.get_trial_checkpoints_paths(
            trial=analysis.get_best_trial(
                'episode_reward_mean',
                mode="max",
                scope="all",
                filter_nan_and_inf=True),
                metric='episode_reward_mean')
        # retrieve the checkpoint path; we only have a single checkpoint, so take the first one
        checkpoint_path = checkpoints[0][0]
        
        return checkpoint_path, analysis
    

    def load(self, path):
        """
        Load a trained RLlib agent from the specified path. Call this before testing a trained agent.
        :param path: Path pointing to the agent's saved checkpoint (only used for RLlib agents)
        """
        self.agent = ray.rllib.agents.ppo.PPOTrainer(config=self.config_model, env="myEnv")
        self.agent.restore(path)
    
    def test(self):
        """Test trained agent for a single episode. Return the episode reward"""
        # we don't need to instantiate the env because we have a brownian motion generator
        env.hedger.verbose = False
        global reward_history
        episode_reward = 0
        total_steps = 0
        rewards = []
        obs = env.reset()
        while True:
            action = self.agent.compute_action(obs)
            obs, reward, done, _ = env.step(action)
            print("done", done)
            episode_reward += reward
            total_steps += 1
            rewards.append(episode_reward)
            print("{}: reward={} action={}".format(total_steps, episode_reward, action))
            reward_history += [episode_reward]
            if env.timestamp % 5 == 0:
                db.plot_rewards(env, reward_history)
            if done:
                break
        
    
if __name__ == "__main__":
    #checkpoint_path = "ppo_crypto_batch30/PPO_myEnv_fd162_00000_0_2021-03-17_13-47-29/checkpoint_25/checkpoint-25"
    training = Training()
    # Train and save 
    #checkpoint_path, results = training.train()
    # Load saved
    #training.load(checkpoint_path)
    # Test loaded
    #training.test()



