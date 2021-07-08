"""
Adopted from Alexey Bakshaev's "Market-making with reinforcement-learning (SAC)" , 2000
Created by Ryan Martin, 2021
"""
import matplotlib.pyplot as plt
from statistics import mean
import dashboard as db

from hedging_env import HedgingEnv, RewardType
import os
import shutil

import ray
from ray import tune
import ray.rllib.agents.sac as sac
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()

def env_creator(env_name):
    if env_name == "HedgeEnv-v0":
        from hedging_env import HedgingEnv as env
    else:
        raise NotImplementedError
    return env
        
env = env_creator("HedgeEnv-v0")
tune.register_env('myEnv', lambda config: env(use_skew=False, reward_type=RewardType.MaxPnl))

class TorchCustomModel(TorchModelV2, nn.Module):
    """Example of a PyTorch custom model that just delegates to a fc-net."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        self.torch_sub_model = TorchFC(obs_space, action_space, num_outputs,
                                       model_config, name)

    def forward(self, input_dict, state, seq_lens):
        input_dict["obs"] = input_dict["obs"].float()
        fc_out, _ = self.torch_sub_model(input_dict, state, seq_lens)
        return fc_out, []

    def value_function(self):
        return torch.reshape(self.torch_sub_model.value_function(), [-1])

class Training(object):
    def __init__(self):
        ray.shutdown()
        ray.init(num_cpus=16, num_gpus=0, ignore_reinit_error=True)
        ModelCatalog.register_custom_model("my_model", TorchCustomModel)
        #ModelCatalog.register_custom_model("attention_net", GTrXLNet)
        self.run = "SAC"
        self.config_model = sac.DEFAULT_CONFIG.copy()
        #self.config_model["Q_model"] = sac.DEFAULT_CONFIG["Q_model"].copy()
        self.config_model["policy_model"] = sac.DEFAULT_CONFIG["policy_model"].copy()
        self.config_model["env"] = "myEnv" 
        self.config_model["horizon"] = 100
        self.config_model["soft_horizon"] = True
        self.config_model["gamma"] = .99
        self.config_model["no_done_at_end"] = True
        self.config_model["tau"] = .003
        self.config_model["n_step"] = 2
        self.config_model["target_entropy"] = "auto"
        self.config_model["target_network_update_freq"] = 32
        self.config_model["num_workers"] = 1  # Run locally.
        self.config_model["twin_q"] = True
        self.config_model["clip_actions"] = True
        self.config_model["normalize_actions"] = True
        self.config_model["learning_starts"] = 256
        self.config_model["prioritized_replay"] = True
        self.config_model["train_batch_size"] = 256
        self.config_model["metrics_smoothing_episodes"] = 5
        self.config_model["_deterministic_loss"] = True
        self.config_model["timesteps_per_iteration"] = 100
            # Use a Beta-distribution instead of a SquashedGaussian for bounded,
            # continuous action spaces (not recommended, for debugging only).
        self.config_model["_use_beta_distribution"] = False
        self.config_model["optimization"]["actor_learning_rate"] = 0.001
        self.config_model["optimization"]["critic_learning_rate"] = 0.001
        self.config_model["optimization"]["entropy_learning_rate"] = 0.003
        self.config_model["framework"]="torch"
        self.stop = {
            "training_iteration": 1000,
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
            ray.rllib.agents.sac.SACTrainer,
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
        self.agent = ray.rllib.agents.sac.SACTrainer(config=self.config_model, env="myEnv")
        self.agent.restore(path)
    
    def test(self):
        """Test trained agent for a single episode. Return the episode reward"""
        # we don't need to instantiate the env because we have a brownian motion generator
        #env.hedger.verbose = False
        env = HedgingEnv(use_skew=False, reward_type=RewardType.MaxPnl)
        reward_history = []
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
    checkpoint_path = "saves/SAC_2021-07-08_04-57-28/SAC_myEnv_fef7c_00000_0_2021-07-08_04-57-29/checkpoint_000100/checkpoint-100"
    training = Training()
    # Train and save 
    checkpoint_path, results = training.train()
    # Load saved
    #training.load(checkpoint_path)
    # Test loaded
    #training.test()



