import numpy as np
from sb3_contrib import RecurrentPPO
from py_bridge_designer.bridge_env import BridgeEnv
from gymnasium.wrappers.time_limit import TimeLimit
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback, ProgressBarCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize, VecVideoRecorder
import wandb
from wandb.integration.sb3 import WandbCallback

SEED = int(np.random.uniform(low=0, high=500000))
print("Seed:", SEED)



def make_env():
    env = BridgeEnv(render_mode="rgb_array")
    env = TimeLimit(env, max_episode_steps=500)
    return env

env = VecNormalize(make_vec_env(make_env, n_envs=8, seed=SEED))

# Config wandb
config = {
    "seed": SEED,
    "render_mode": "rgb_array",
    "time_limit_max_episode_steps": 500,
    "parallel_traning_n_envs": 8,
    "parallel_eval_n_envs": 4,
    "n_eval_episode": 10,
    "eval_freq": 5000,
    "policy_type": "MlpLstmPolicy",
    "total_timesteps": 100000,
    "env_id": "BridgeEnv",
    "learning_rate": 0.0001,
    "batch_size": 128,
    "max_grad_norm": 1,
    "gae_lambda": 0.98,
    "gamma": 0.98,
    "policy_kwargs": {
        "net_arch": {"vf": 64, "pi": 0},
        "lstm_hidden_size": 64,
        "ortho_init": False,
        "enable_critic_lstm": True,
    }
}

# Init wandb
run = wandb.init(
    project="thesis",
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    monitor_gym=True,  # auto-upload the videos of agents playing the game
    save_code=True,  # optional
)

# Create the callbacks
eval_callback = EvalCallback(
    VecNormalize(make_vec_env(make_env, n_envs=4), training=False, norm_reward=False),
    n_eval_episodes=10,
    eval_freq=5000
)
progress_bar = ProgressBarCallback()

wandb_callback = WandbCallback(
    gradient_save_freq=1000,
    model_save_path=f"models/{run.id}",
    verbose=2,
)

callback = CallbackList([eval_callback, progress_bar, wandb_callback])

# Record Videos
env = VecVideoRecorder(
    env,
    f"videos/{run.id}",
    record_video_trigger=lambda x: x % 2000 == 0,
    video_length=200,
)

# Set the model
model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        learning_rate=0.0001,
        verbose=1,
        batch_size=128,
        seed=SEED,
        max_grad_norm=1,
        gae_lambda=0.98,
        gamma=0.98,
        policy_kwargs=dict(
            net_arch=dict(vf=[64], pi=[]),
            lstm_hidden_size=64,
            ortho_init=False,
            enable_critic_lstm=True,
        ),
        tensorboard_log=f"runs/{run.id}"
)

model.learn(total_timesteps=100000, callback=callbacks)
evaluate_policy(model, env)