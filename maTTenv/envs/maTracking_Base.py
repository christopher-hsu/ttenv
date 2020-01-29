import os, copy, pdb
import numpy as np
from numpy import linalg as LA
import gym
from gym import spaces, logger

from maTTenv.maps import map_utils
import maTTenv.env_utils as util 
from maTTenv.metadata import METADATA


class maTrackingBase(gym.Env):
    def __init__(self, num_agents=2, num_targets=1, map_name='empty',
                        is_training=True, known_noise=True, **kwargs):
        self.seed()
        self.id = 'maTracking-base'
        self.action_space = spaces.Discrete(len(METADATA['action_v']) * \
                                                len(METADATA['action_w']))
        self.action_map = {}
        for (i,v) in enumerate(METADATA['action_v']):
            for (j,w) in enumerate(METADATA['action_w']):
                self.action_map[len(METADATA['action_w'])*i+j] = (v,w)
        assert(len(self.action_map.keys())==self.action_space.n)

        self.agent_dim = 3
        self.target_dim = 2
        self.num_agents = num_agents
        self.num_targets = num_targets
        self.viewer = None
        self.is_training = is_training

        self.sampling_period = 0.5 # sec
        self.q = q if self.num_targets==1 else 0.1*q
        self.sensor_r_sd = METADATA['sensor_r_sd']
        self.sensor_b_sd = METADATA['sensor_b_sd']
        self.sensor_r = METADATA['sensor_r']
        self.fov = METADATA['fov']
        map_dir_path = '/'.join(map_utils.__file__.split('/')[:-1])
        self.MAP = map_utils.GridMap(
            map_path=os.path.join(map_dir_path, hyp['map_name']), 
            r_max = self.sensor_r, fov = self.fov/180.0*np.pi,
            margin2wall = METADATA['margin2wall'])
        # LIMITS
        self.limit = {} # 0: low, 1:high
        self.limit['agent'] = [np.concatenate((self.MAP.mapmin,[-np.pi])), np.concatenate((self.MAP.mapmax, [np.pi]))]
        self.limit['target'] = [self.MAP.mapmin, self.MAP.mapmax]
        self.limit['state'] = [np.concatenate(([0.0, -np.pi, -50.0, 0.0]*self.num_targets, [0.0, -np.pi ])),
                               np.concatenate(([600.0, np.pi, 50.0, 2.0]*self.num_targets, [self.sensor_r, np.pi]))]
        self.observation_space = spaces.Box(self.limit['state'][0], self.limit['state'][1], dtype=np.float32)

        self.agent_init_pos =  np.array([self.MAP.origin[0], self.MAP.origin[1], 0.0])
        self.target_init_pos = np.array(self.MAP.origin)
        self.target_init_cov = METADATA['target_init_cov']
        self.target_noise_cov = METADATA['const_q'] * self.sampling_period**3/3*np.eye(self.target_dim)
        if known_noise:
            self.target_true_noise_sd = self.target_noise_cov
        else:
            self.target_true_noise_sd = METADATA['const_q_true']*np.eye(2)
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

    def gen_rand_pose(self, o_xy, c_theta, min_lin_dist, max_lin_dist, min_ang_dist, max_ang_dist):
        """Genertes random position and yaw.
        Parameters
        --------
        o_xy : xy position of a point in the global frame which we compute a distance from.
        c_theta : angular position of a point in the global frame which we compute an angular distance from.
        min_lin_dist : the minimum linear distance from o_xy to a sample point.
        max_lin_dist : the maximum linear distance from o_xy to a sample point.
        min_ang_dist : the minimum angular distance (counter clockwise direction) from c_theta to a sample point.
        max_ang_dist : the maximum angular distance (counter clockwise direction) from c_theta to a sample point.
        """
        if max_ang_dist < min_ang_dist:
            max_ang_dist += 2*np.pi
        rand_ang = util.wrap_around(np.random.rand() * \
                        (max_ang_dist - min_ang_dist) + min_ang_dist + c_theta)

        rand_r = np.random.rand() * (max_lin_dist - min_lin_dist) + min_lin_dist
        rand_xy = np.array([rand_r*np.cos(rand_ang), rand_r*np.sin(rand_ang)]) + o_xy
        is_valid = not(map_utils.is_collision(self.MAP, rand_xy))
        return is_valid, [rand_xy[0], rand_xy[1], rand_ang]

    def get_init_pose(self, init_pose_list=[], **kwargs):
        """Generates initial positions for the agent, targets, and target beliefs.
        Parameters
        ---------
        init_pose_list : a list of dictionaries with pre-defined initial positions.
        lin_dist_range : a tuple of the minimum and maximum distance of a target
                        and a belief target from the agent.
        ang_dist_range_target : a tuple of the minimum and maximum angular
                            distance (counter clockwise) of a target from the
                            agent. -pi <= x <= pi
        ang_dist_range_belief : a tuple of the minimum and maximum angular
                            distance (counter clockwise) of a belief from the
                            agent. -pi <= x <= pi
        blocked : True if there is an obstacle between a target and the agent.
        """
        if init_pose_list:
            self.reset_num += 1
            return init_pose_list[self.reset_num-1]
        else:
            return self.get_init_pose_random(**kwargs)

    def get_init_pose_random(self,
                            lin_dist_range=(METADATA['init_distance_min'], METADATA['init_distance_max']),
                            ang_dist_range_target=(-np.pi, np.pi),
                            ang_dist_range_belief=(-np.pi, np.pi),
                            blocked=False,
                            **kwargs):
        is_agent_valid = False
        while(not is_agent_valid):
            init_pose = {}
            if self.MAP.map is None:
                if blocked:
                    raise ValueError('Unable to find a blocked initial condition. There is no obstacle in this map.')
                a_init = self.agent_init_pos[:2]
                is_agent_valid = True
            else:
                while(not is_agent_valid):
                    a_init = np.random.random((2,)) * (self.MAP.mapmax-self.MAP.mapmin) + self.MAP.mapmin
                    is_agent_valid = not(map_utils.is_collision(self.MAP, a_init))

            init_pose['agent'] = [a_init[0], a_init[1], np.random.random() * 2 * np.pi - np.pi]
            init_pose['targets'], init_pose['belief_targets'] = [], []
            for i in range(self.num_targets):
                count, is_target_valid = 0, False
                while(not is_target_valid):
                    is_target_valid, init_pose_target = self.gen_rand_pose(
                        init_pose['agent'][:2], init_pose['agent'][2],
                        lin_dist_range[0], lin_dist_range[1],
                        ang_dist_range_target[0], ang_dist_range_target[1])
                    is_blocked = map_utils.is_blocked(self.MAP, init_pose['agent'][:2], init_pose_target[:2])
                    if is_target_valid:
                        is_target_valid = (blocked == is_blocked)
                    count += 1
                    if count > 100:
                        is_agent_valid = False
                        break
                init_pose['targets'].append(init_pose_target)

                count, is_belief_valid, init_pose_belief = 0, False, np.zeros((2,))
                while((not is_belief_valid) and is_target_valid):
                    is_belief_valid, init_pose_belief = self.gen_rand_pose(
                        init_pose['agent'][:2], init_pose['targets'][i][2],
                        lin_dist_range[0], lin_dist_range[1],
                        ang_dist_range_belief[0], ang_dist_range_belief[1])
                    count += 1
                    if count > 100:
                        is_agent_valid = False
                        break
                init_pose['belief_targets'].append(init_pose_belief)
        return init_pose