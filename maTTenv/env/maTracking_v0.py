import os, copy, pdb
import numpy as np
from numpy import linalg as LA
from gym import spaces, logger
from maTTenv.maps import map_utils
import maTTenv.utils as util 
from maTTenv.agent_models import *
from maTTenv.belief_tracker import KFbelief
from maTTenv.metadata import METADATA
from maTTenv.env.maTracking_Base import maTrackingBase

"""
[Variables]

d: radial coordinate of a belief target in the learner frame
alpha : angular coordinate of a belief target in the learner frame
ddot : radial velocity of a belief target in the learner frame
alphadot : angular velocity of a belief target in the learner frame
Sigma : Covariance of a belief target
o_d : linear distance to the closet obstacle point
o_alpha : angular distance to the closet obstacle point

[Environment Description]

maTrackingEnv0 : Static Target model + noise - No Velocity Estimate
    obs_dict: [d, alpha, logdet(Sigma)] * nb_targets , [o_d, o_alpha]
    Agent: SE2 [x,y,orientation]
    Target: Static [x,y] + noise
    Belief Target: KF, Estimate only x and y
"""

class maTrackingEnv0(maTrackingBase):

    def __init__(self, num_agents=2, num_targets=2, map_name='empty', 
                        is_training=True, known_noise=True, **kwargs):
        super().__init__(num_agents=num_agents, num_targets=num_targets, 
                        map_name=map_name, is_training=is_training)
       
        self.id = 'maTracking-v0'
        self.agent_dim = 3
        self.target_dim = 2
       
        # Build a robot 
        self.setup_agents()
        # Build a target
        self.setup_targets()
        self.setup_belief_targets()

    def setup_agents(self):
        self.agents = [AgentSE2(agent_id = 'agent-' + str(i), 
                        dim=self.agent_dim, sampling_period=self.sampling_period, limit=self.limit['agent'], 
                        collision_func=lambda x: map_utils.is_collision(self.MAP, x))
                        for i in range(self.num_agents)]

    def setup_targets(self):
        self.targets = [AgentDoubleInt2D(agent_id = 'target-' + str(i),
                        dim=self.target_dim, sampling_period=self.sampling_period, limit=self.limit['target'],
                        collision_func=lambda x: map_utils.is_collision(self.MAP, x),
                        A=self.targetA, W=self.target_true_noise_sd) 
                        for i in range(self.num_targets)]

    def setup_belief_targets(self):
        self.belief_targets = [KFbelief(agent_id = 'agent-' + str(i),
                        dim=self.target_dim, limit=self.limit['target'], A=self.targetA,
                        W=self.target_noise_cov, obs_noise_func=self.observation_noise, 
                        collision_func=lambda x: map_utils.is_collision(self.MAP, x))
                        for i in range(self.num_targets)]

    def reset(self, **kwargs):
        """
        Agents are given random positions in the map, targets are given random positions near a random agent.
        Return a state dict with agent ids (keys) and target state info
        """
        obs_dict = {}
        init_pose = self.get_init_pose(**kwargs)
        # Initialize agents
        for ii in range(self.num_agents):
            self.agents[ii].reset(init_pose['agents'][ii])
            obs_dict[self.agents[ii].agent_id] = []
        # Initialize targets and beliefs
        for jj in range(self.num_targets):
            self.belief_targets[jj].reset(
                        init_state=np.array(init_pose['belief_targets'][jj][:self.target_dim]),
                        init_cov=self.target_init_cov)
            self.targets[jj].reset(np.array(init_pose['targets'][jj][:self.target_dim]))
            #For each agent calculate belief of all targets
            for kk in range(self.num_agents):
                r, alpha = util.relative_distance_polar(self.belief_targets[jj].state[:2],
                                     xy_base=self.agents[kk].state[:2], 
                                     theta_base=self.agents[kk].state[2])
                logdetcov = np.log(LA.det(self.belief_targets[jj].cov))
                obs_dict[self.agents[kk].agent_id].extend([r, alpha, logdetcov, 0.0])
        for agent_id in obs_dict:
            obs_dict[agent_id].extend([self.sensor_r, np.pi])
        return obs_dict

    def step(self, action_dict):
        obs_dict = {}
        reward_dict = {}
        done_dict = {'__all__':False}
        info_dict = {}

        # Targets move (t -> t+1)
        for n in range(self.num_targets):
            self.targets[n].update() 
        # Agents move (t -> t+1) and observe the targets
        for ii, agent_id in enumerate(action_dict):
            obs_dict[self.agents[ii].agent_id] = []
            reward_dict[self.agents[ii].agent_id] = []
            done_dict[self.agents[ii].agent_id] = []

            action_vw = self.action_map[action_dict[agent_id]]
            _ = self.agents[ii].update(action_vw, [t.state[:2] for t in self.targets])
            
            observed = []
            for jj in range(self.num_targets):
                # Observe
                obs = self.observation(self.targets[jj], self.agents[ii])
                observed.append(obs[0])
                self.belief_targets[jj].predict() # Belief state at t+1
                if obs[0]: # if observed, update the target belief.
                    self.belief_targets[jj].update(obs[1], self.agents[ii].state)

            obstacles_pt = map_utils.get_closest_obstacle(self.MAP, self.agents[ii].state)
            #Reward after each agent step
            # reward, done, mean_nlogdetcov = self.get_reward(obstacles_pt, observed, self.is_training)
            # reward_dict[agent_id], done_dict[agent_id], info_dict[agent_id] = reward, done, mean_nlogdetcov

            if obstacles_pt is None:
                obstacles_pt = (self.sensor_r, np.pi)
            for kk in range(self.num_targets):
                r_b, alpha_b = util.relative_distance_polar(self.belief_targets[kk].state[:2],
                                                xy_base=self.agents[ii].state[:2], 
                                                theta_base=self.agents[ii].state[-1])
                obs_dict[agent_id].extend([r_b, alpha_b, 
                                        np.log(LA.det(self.belief_targets[kk].cov)), float(observed[kk])])
            obs_dict[agent_id].extend([obstacles_pt[0], obstacles_pt[1]])
        # Get all rewards after all agents and targets move (t -> t+1)
        reward, done, mean_nlogdetcov = self.get_reward(obstacles_pt, observed, self.is_training)
        reward_dict['__all__'], done_dict['__all__'], info_dict['mean_nlogdetcov'] = reward, done, mean_nlogdetcov
        return obs_dict, reward_dict, done_dict, info_dict