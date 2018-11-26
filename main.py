import multiprocessing
import os

from stable_baselines.common.policies import MlpLnLstmPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import A2C

from stable_baseline_wrapper import make_or_create_envs

# specify model data
model_type = A2C
policy = MlpLnLstmPolicy
env_id = 'HandManipulatePen-v0'
model_name = f'{model_type.__name__}_{policy.__name__}_{env_id}'
log = f'{model_name}_log'
overwrite_previous_model = False
retrain = False

# get number of CPUs
n_cpu = multiprocessing.cpu_count()

# get pickled environments if they exist, else create them
envs = make_or_create_envs(env_id, n_cpu, force_create=False)

# initialize multiprocessor environment
env = SubprocVecEnv([lambda: envs[i] for i in range(n_cpu)])

if overwrite_previous_model or not os.path.exists(f'{model_name}.pkl'):
    model = model_type(policy, env, verbose=1, tensorboard_log=log)
    print(f'{model_name} created...')
    model.save(model_name)
    del model
else:
    print(f'Loading {model_name}...')

model = model_type.load(model_name, env)

if retrain:
    print('Training model from last checkpoint')
    model.learn(total_timesteps=10000)

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
