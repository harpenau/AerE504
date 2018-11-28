import gym
import numpy as np

from gym.spaces import Box, Dict
from stable_baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines.common.policies import LstmPolicy


class BaselineWrapper(gym.Wrapper):
    '''Wrapper for a gym.core.GoalEnv so that it interfaces with
    stable_baselines' API.

    This wrapper flattens the observation, overrides the comuputed
    reward and trims incoming actions before inputting them into self.step.'''

    def __init__(self, env_id):
        '''Makes a wrapped gym.Env as specifiec by env_id.'''
        super().__init__(gym.make(env_id))
        self.observation_space = self._observation_space()
        self.action_space = self._action_space()

    def _action_space(self):
        '''Get some information about this env and return the action_space.'''
        self.continuous_action_space = isinstance(self.env.action_space, Box)
        return self.env.action_space

    def _observation_space(self):
        '''Get some information about this env. Return the untouched
        observation_space if thie is env is not a GoalEnv, otherwise
        return a Box with the same characteristics as the 3 Boxes
        that compose this model.'''
        self.goal_env = isinstance(self.env.observation_space, Dict)
        if self.goal_env:
            space = list(self.env.observation_space.spaces.values())
            low = np.append(space[-1].low, [s.low for s in space[:-1]])
            high = np.append(space[-1].high, [s.high for s in space[:-1]])
            return Box(low, high, dtype=high.dtype)
        else:
            return self.env.observation_space

    def reset(self, **kwargs):
        '''Return the flattened observation.'''
        self.epsilon = .1
        return self.observation(self.env.reset(**kwargs))

    def step(self, action):
        '''Execute given action and return overridden observation,
        overridden reward, done, and info'''
        observation, reward, done, info = self.env.step(self.action(action))
        if done:
            self.reset()
        return (self.observation(observation), reward, done, info)

    def action(self, action):
        '''Run epsilon-greedy algorithm where epsilon is derived from
        the best action since the last reset. Trim and action if
        necessary.'''
        if self.goal_env and np.random.random() < self.epsilon:  # .1
            return self.action_space.sample()

        if self.continuous_action_space:
            low, high = self.action_space.low, self.action_space.high
            for i, _ in enumerate(action):
                if low[i] >= action[i]:
                    action[i] = low[i]
                if action[i] >= high[i]:
                    action[i] = high[i]
        return action

    def observation(self, observation):
        '''If this env is a GoalEnv, return the flattened observation,
        otherwise return the original observation.'''
        if self.goal_env:
            observation_, achieved_goal, desired_goal = observation.values()
            return np.append(observation_, [achieved_goal, desired_goal])
        else:
            return observation

    def reward(self, reward, observation):
        '''If this env is a GoalEnv, calculate the overridden reward,
        else return the reward.'''
        if self.goal_env:
            _, achieved, desired = observation.values()
#             rms = np.sqrt(np.mean(np.square(desired - achieved)))
            reward += sum(1 - abs(desired - achieved))
            return reward
        else:
            return reward


class VectorEnvironmentWrapper(SubprocVecEnv):
    '''Wrapper for SubprocVecEnv to alter render functionality'''

    def __init__(self, env_id, n_env):
        '''Initializes object from super class.'''
        super().__init__([lambda: BaselineWrapper(env_id) for _ in range(n_env)])

    def render(self, number=1, tiled=False, mode='human', *args, **kwargs):
        '''Override render to add functionality of choosing many env
        should be displayed on screen. Also add boolean as to whether
        or not a tiled image should be put together and displayed.'''
        number = len(self.remotes) if number > len(self.remotes) else number
        for i in range(number):
            # gather images from subprocesses
            # `mode` will be taken into account later
            self.remotes[i].send(('render', (args, {'mode': 'rgb_array', **kwargs})))
        imgs = [self.remotes[i].recv() for i in range(number)]
        if tiled:
            # Create a big image by tiling images from subprocesses
            from stable_baselines.common.tile_images import tile_images
            bigimg = tile_images(imgs)
            if mode == 'human':
                import cv2
                cv2.imshow('vecenv', bigimg[:, :, ::-1])
                cv2.waitKey(1)
            elif mode == 'rgb_array':
                return bigimg


class LargeMlpLnLstmPolicy(LstmPolicy):
    """
    Policy object that implements actor critic, using a layer normalized LSTMs with a MLP feature extraction

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param n_lstm: (int) The number of LSTM cells (for recurrent policies)
    :param reuse: (bool) If the policy is reusable or not
    :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **_kwargs):
        super().__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=512, reuse=reuse,
                         layers=[128, 128], layer_norm=True, feature_extraction="mlp", **_kwargs)


if __name__ == '__main__':
    env = BaselineWrapper('HandManipulatePen-v0')

    env.reset()

    while True:
        action = env.action_space.sample()

        observation, reward, done, info = env.step(action)

        print(reward, env.epsilon)

        env.render()
