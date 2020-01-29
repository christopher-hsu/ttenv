import os, copy, pdb

import numpy as np
from numpy import linalg as LA

from gym import spaces, logger

from envs.maps import map_utils
import envs.env_utils as util 
from envs.maTracking.metadata import *

from ray.rllib.env import MultiAgentEnv


class maTrackingBase(MultiAgentEnv):
    def __init__(self, hyp, agent_dim, target_dim, q_true = 0.01, q = 0.01, is_training=True, known_noise=True):

        self.hyp = hyp
        self.action_space = spaces.Discrete(12)
        self.action_map = {}
        for (i,v) in enumerate([3,2,1,0]):
            for (j,w) in enumerate([np.pi/2, 0, -np.pi/2]):
                self.action_map[3*i+j] = (v,w)
        self.agent_dim = agent_dim
        self.target_dim = target_dim
        self.num_agents = hyp['num_agents']
        self.num_targets = hyp['num_targets']
        self.viewer = None
        self.is_training = is_training

        self.sampling_period = 0.5 # sec
        self.q = q if self.num_targets==1 else 0.1*q
        self.sensor_r_sd = METADATA['sensor_r_sd']
        self.sensor_b_sd = METADATA['sensor_b_sd']
        self.sensor_r = METADATA['sensor_r']
        self.fov = METADATA['fov']

        map_dir_path = '/'.join(map_utils.__file__.split('/')[:-1])

        self.MAP = map_utils.GridMap(map_path=os.path.join(map_dir_path, hyp['map_name']), 
                                        r_max = self.sensor_r, fov = self.fov/180.0*np.pi)
        # LIMITS
        self.limit = {} # 0: low, 1:high
        self.limit['agent'] = [np.concatenate((self.MAP.mapmin,[-np.pi])), np.concatenate((self.MAP.mapmax, [np.pi]))]
        self.limit['target'] = [self.MAP.mapmin, self.MAP.mapmax]
        self.limit['state'] = [np.concatenate(([0.0, -np.pi, -50.0, 0.0]*self.num_targets, [0.0, -np.pi ])),
                               np.concatenate(([600.0, np.pi, 50.0, 2.0]*self.num_targets, [self.sensor_r, np.pi]))]
        self.observation_space = spaces.Box(self.limit['state'][0], self.limit['state'][1], dtype=np.float32)

        self.agent_init_pos =  np.array([self.MAP.origin[0], self.MAP.origin[1], 0.0])

        self.target_init_pos = np.array(self.MAP.origin)
        self.target_init_cov = TARGET_INIT_COV
        self.target_noise_cov = self.q * self.sampling_period**3/3*np.eye(self.target_dim)
        if known_noise:
            self.target_true_noise_sd = self.target_noise_cov
        else:
            self.target_true_noise_sd = q_true*np.eye(2)
        self.targetA=np.eye(self.target_dim)

        #needed for gym/core.py wrappers
        self.metadata = {'render.modes': []}
        self.reward_range = (-float('inf'), float('inf'))
        self.spec = None

    def setup_agents(self):
        """Construct all the agents for the environment"""
        raise NotImplementedError

    def setup_targets(self):
        """Construct all the targets for the environment"""
        raise NotImplementedError

    def setup_belief_targets(self):
        """Construct all the target beliefs for the environment"""
        raise NotImplementedError

    def reset(self, init_random=True):
        """Reset the state of the environment."""
        raise NotImplementedError

    def step(self, action_dict):
        """Takes in dict of action and coverts them to map updates (obs, rewards)"""
        raise NotImplementedError

    def observation(self, target, agent):
        r, alpha, _ = util.relative_measure(target.state, agent.state) # True Value     
        observed = (r <= self.sensor_r) \
                    & (abs(alpha) <= self.fov/2/180*np.pi) \
                    & (not(map_utils.is_blocked(self.MAP, agent.state, target.state)))
        z = None
        if observed:
            z = np.array([r, alpha])
            z += np.random.multivariate_normal(np.zeros(2,), self.observation_noise(z))
        return observed, z

    def observation_noise(self, z):
        obs_noise_cov = np.array([[self.sensor_r_sd * self.sensor_r_sd, 0.0], #z[0]/self.sensor_r * self.sensor_r_sd, 0.0], 
                                [0.0, self.sensor_b_sd * self.sensor_b_sd]])
        return obs_noise_cov

    def get_reward(self, obstacles_pt, observed, tot_observed, is_training=True):
        if obstacles_pt is None:
            penalty = 0.0
        else:
            penalty = 1./max(1.0, obstacles_pt[0]**2)/self.num_agents

        if sum(observed) == 0:
            reward = - penalty - 1
        else:
            detcov = [LA.det(b_target.cov) for b_target in self.belief_targets]
            reward = - 0.1 * np.log(np.mean(detcov) + np.std(detcov)) - penalty
            # logdetcov = [-np.log(LA.det(b_target.cov)) for b_target in self.belief_targets]
            # reward = (np.mean(logdetcov)+np.std(logdetcov))/self.num_agents
            reward = max(0.0, reward) + np.mean(observed)
            # reward = reward + np.mean(tot_observed)
        test_reward = None

        if not(is_training):
            logdetcov = [np.log(LA.det(b_target.cov)) for b_target in self.belief_targets]              
            test_reward = -np.mean(logdetcov)

        done = False
        return reward, done, test_reward