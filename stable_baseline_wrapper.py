import os
import pickle
import gym
import numpy as np

from gym.spaces import Box, Dict


class BaselineWrapper(gym.Wrapper):

    def __init__(self, env_id):
        super().__init__(gym.make(env_id))
        self.observation_space = self._observation_space()
        self.action_space = self._action_space()

    def _action_space(self):
        self.continuous_action_space = isinstance(self.env.action_space, Box)
        return self.env.action_space

    def _observation_space(self):
        self.goal_env = isinstance(self.env.observation_space, Dict)
        if self.goal_env:
            space = list(self.env.observation_space.spaces.values())
            low = np.append(space[-1].low, [s.low for s in space[:-1]])
            high = np.append(space[-1].high, [s.high for s in space[:-1]])
            return Box(low, high, dtype=high.dtype)
        else:
            return self.env.observation_space

    def reset(self, **kwargs):
        return self.observation(self.env.reset(**kwargs))

    def step(self, action):
        observation, reward, done, info = self.env.step(self.action(action))
        return (self.observation(observation),
                self.reward(reward, observation), done, info)

    def action(self, action):
        if action.shape != self.action_space.shape:
            action = action.reshape(self.action_space.shape)
        if self.continuous_action_space:
            low, high = self.env.action_space.low, self.env.action_space.high
            for i, _ in enumerate(action):
                if low[i] >= action[i]:
                    action[i] = low[i]
                if action[i] >= high[i]:
                    action[i] = high[i]
        return action

    def observation(self, observation):
        if self.goal_env:
            observation_, achieved_goal, desired_goal = observation.values()
            return np.append(observation_, [achieved_goal, desired_goal])
        else:
            return observation

    def reward(self, reward, observation):
        if self.goal_env:
            _, achieved, desired = observation.values()
            return 1 - (np.square(desired - achieved)).mean(axis=0) + reward
        else:
            return reward


def make_or_create_envs(env_id, how_many, force_create=False):
    pickled_envs_file = f'{how_many}_pickled_{env_id}_envs'
    if os.path.exists(pickled_envs_file) and not force_create:
        envs = pickle.load(open(pickled_envs_file, 'rb'))
        print(f'{how_many} {env_id} environments have been loaded.')
    else:
        envs = [BaselineWrapper(env_id) for _ in range(how_many)]
        pickle.dump(envs, open(pickled_envs_file, 'wb'))
        print(f'{how_many} {env_id} new environments have been created and pickled.')
    return envs


if __name__ == '__main__':
    env = BaselineWrapper('HandManipulatePen-v0')

    print(env.unwrapped)
