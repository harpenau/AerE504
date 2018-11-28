import os
import multiprocessing

from stable_baselines.a2c.a2c import A2C
from stable_baselines.common.policies import MlpLnLstmPolicy
from stable_baselines.common.vec_env.vec_normalize import VecNormalize

from stable_baseline_wrapper import VectorEnvironmentWrapper


# specify model type, policy type, define model name
model_type, policy, model = A2C, MlpLnLstmPolicy, None
env_id = 'HandManipulatePen-v0'
model_name = f'{model_type.__name__}_{policy.__name__}_{env_id}'
log = f'{model_name}_log'

# specify model parameters
model_kwargs = {'gamma': 0.99,              # default: 0.99
                'n_steps': 5,               # default: 5
                'vf_coef': 0.25,            # default: 0.25
                'ent_coef': 0.01,           # default: 0.01
                'max_grad_norm': 0.5,       # default: 0.5
                'learning_rate': 2e-4,      # default: 7e-4
                'alpha': 0.99,              # default: 0.99
                'epsilon': 1e-5,            # default: 1e-5
                'lr_schedule': 'linear',    # default: 'linear'
                'verbose': 1,               # default: 0
                'tensorboard_log': log,     # default: None
                '_init_setup_model': True}  # default: True

# specify whether or not to try to create a new model and/or train it
overwrite_previous_model, train = True, True

# get number of CPUs
n_cpu = multiprocessing.cpu_count()

# initialize multiprocessor environment
env = VecNormalize(VectorEnvironmentWrapper(env_id, n_cpu))

# create/load the model, maybe train it
if overwrite_previous_model or not os.path.exists(f'{model_name}.pkl'):
    model = model_type(policy, env, **model_kwargs)
    print(f'{model_name} created...')
if not model:
    model = model_type.load(model_name, env)
    print(f'Loading {model_name}...')
if train:
    print('Training model from last checkpoint')
    model.learn(total_timesteps=100000, log_interval=100)
    model.save(model_name)

# render the env
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    print(rewards[0])
    env.render()
