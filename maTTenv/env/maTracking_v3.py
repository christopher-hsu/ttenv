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
Target Tracking Environments for Reinforcement Learning.
[Variables]

d: radial coordinate of a belief target or agent in the learner frame
alpha : angular coordinate of a belief target or agent in the learner frame
ddot : radial velocity of a belief target or agent in the learner frame
alphadot : angular velocity of a belief target or agent in the learner frame
Sigma : Covariance of a belief target
o_d : linear distance to the closet obstacle point
o_alpha : angular distance to the closet obstacle point

[Environment Description]

maTargetTrackingEnv2 : Agents locations included in observation state
    Double Integrator Target model with KF belief tracker
    obs state:  [d, alpha, ddot, alphadot, logdet(Sigma), observed] * nb_targets,
                [o_d, o_alpha],
                [d, alpha] * nb_agents-1
    Agent : SE2 model, [x,y,theta]
    Target : Double Integrator model, [x,y,xdot,ydot]
    Belief Target : KF, Double Integrator model
    global state: [d, alpha] * (nb_targets+nb_agents) in ref to origin

>>>print(full_state)
{
    'agent-0':
    {
            'obs':[obs state]
            'state':[global state]
    },
    'agent-1':{}
}
"""

class maTrackingEnv3(maTrackingBase):

    def __init__(self, num_agents=2, num_targets=1, map_name='empty', 
                        is_training=True, known_noise=True, **kwargs):
        super().__init__(num_agents=num_agents, num_targets=num_targets, 
                        map_name=map_name, is_training=is_training)

        self.id = 'maTracking-v3'
        self.agent_dim = 3
        self.target_dim = 4
        self.target_init_vel = METADATA['target_init_vel']*np.ones((2,))
        # LIMIT
        self.limit = {} # 0: low, 1:highs
        self.limit['agent'] = [np.concatenate((self.MAP.mapmin,[-np.pi])), np.concatenate((self.MAP.mapmax, [np.pi]))]
        self.limit['target'] = [np.concatenate((self.MAP.mapmin,[-METADATA['target_vel_limit'], -METADATA['target_vel_limit']])),
                                np.concatenate((self.MAP.mapmax, [METADATA['target_vel_limit'], METADATA['target_vel_limit']]))]
        rel_vel_limit = METADATA['target_vel_limit'] + METADATA['action_v'][0] # Maximum relative speed
        self.limit['state'] = [np.concatenate(([0.0, -np.pi, -rel_vel_limit, -10*np.pi, -50.0, 0.0]*self.num_targets, 
                                np.concatenate(([0.0, -np.pi ],
                                [0.0, -np.pi]*(self.num_agents-1))))),
                                np.concatenate(([600.0, np.pi, rel_vel_limit, 10*np.pi,  50.0, 2.0]*self.num_targets, 
                                np.concatenate(([self.sensor_r, np.pi],
                                [600.0, np.pi]*(self.num_agents-1)))))]
        self.limit['global'] = [np.concatenate((self.limit['state'][0],np.array([0.0,-np.pi]*(self.num_targets+self.num_agents)))),
                                np.concatenate((self.limit['state'][1],np.array([600.0,np.pi]*(self.num_targets+self.num_agents))))]
        self.observation_space = spaces.Box(self.limit['state'][0], self.limit['state'][1], dtype=np.float32)
        self.global_space = spaces.Box(self.limit['global'][0], self.limit['global'][1], dtype=np.float32)
        self.targetA = np.concatenate((np.concatenate((np.eye(2), self.sampling_period*np.eye(2)), axis=1), 
                                        [[0,0,1,0],[0,0,0,1]]))
        self.target_noise_cov = METADATA['const_q'] * np.concatenate((
                        np.concatenate((self.sampling_period**3/3*np.eye(2), self.sampling_period**2/2*np.eye(2)), axis=1),
                        np.concatenate((self.sampling_period**2/2*np.eye(2), self.sampling_period*np.eye(2)),axis=1) ))
        if known_noise:
            self.target_true_noise_sd = self.target_noise_cov
        else:
            self.target_true_noise_sd = METADATA['const_q_true'] * np.concatenate((
                        np.concatenate((self.sampling_period**2/2*np.eye(2), self.sampling_period/2*np.eye(2)), axis=1),
                        np.concatenate((self.sampling_period/2*np.eye(2), self.sampling_period*np.eye(2)),axis=1) ))

        # Build a robot 
        self.setup_agents()
        # Build a target
        self.setup_targets()
        self.setup_belief_targets()

    def setup_agents(self):
        self.agents = [AgentSE2(agent_id = 'agent-' + str(i), 
                        dim=self.agent_dim, sampling_period=self.sampling_period, 
                        limit=self.limit['agent'], 
                        collision_func=lambda x: map_utils.is_collision(self.MAP, x))
                        for i in range(self.num_agents)]

    def setup_targets(self):
        self.targets = [AgentDoubleInt2D(agent_id = 'agent-' + str(i),
                        dim=self.target_dim, sampling_period=self.sampling_period, 
                        limit=self.limit['target'],
                        collision_func=lambda x: map_utils.is_collision(self.MAP, x),
                        A=self.targetA, W=self.target_true_noise_sd) 
                        for i in range(self.num_targets)]

    def setup_belief_targets(self):
        self.belief_targets = [KFbelief(agent_id = 'agent-' + str(i),
                        dim=self.target_dim, limit=self.limit['target'], A=self.targetA,
                        W=self.target_noise_cov, obs_noise_func=self.observation_noise, 
                        collision_func=lambda x: map_utils.is_collision(self.MAP, x))
                        for i in range(self.num_targets)]

    def reset(self,**kwargs):
        """
        Agents are given random positions in the map, targets are given random positions near a random agent.
        Return a full state dict with agent ids (keys) that refer to their observation and global state
        """
        obs_dict = {}
        locations = []
        global_state = {}
        full_state = {}

        init_pose = self.get_init_pose(**kwargs)
        # Initialize agents
        for ii in range(self.num_agents):
            self.agents[ii].reset(init_pose['agents'][ii])
            obs_dict[self.agents[ii].agent_id] = []
        # Initialize targets and beliefs
        for jj in range(self.num_targets):
            self.belief_targets[jj].reset(
                        init_state=np.concatenate((init_pose['belief_targets'][jj][:2], np.zeros(2))),
                        init_cov=self.target_init_cov)
            self.targets[jj].reset(np.concatenate((init_pose['targets'][jj][:2], self.target_init_vel)))
            locations.append(self.targets[jj].state[:2])
            #For each agent calculate belief of all targets
            for kk in range(self.num_agents):
                r, alpha = util.relative_distance_polar(self.belief_targets[jj].state[:2],
                                            xy_base=self.agents[kk].state[:2], 
                                            theta_base=self.agents[kk].state[2])
                logdetcov = np.log(LA.det(self.belief_targets[jj].cov))
                obs_dict[self.agents[kk].agent_id].extend([r, alpha, 0.0, 0.0, logdetcov, 0.0])
        # All targets and agents locations relative to map origin (targets then agents)
        for n in range(self.num_agents):
            locations.append(self.agents[n].state[:2])
        global_state = util.global_relative_measure(np.array(locations), self.MAP.origin)
        # Full state dict         
        for m, agent_id in enumerate(obs_dict):
            obs_dict[agent_id].extend([self.sensor_r, np.pi])
            # Relative location and past action of all other agents
            for p, ids in enumerate(obs_dict):
                if agent_id != ids:
                    r, alpha = util.relative_distance_polar(np.array(self.agents[p].state[:2]),
                                            xy_base=self.agents[m].state[:2], 
                                            theta_base=self.agents[m].state[2])
                    obs_dict[agent_id].extend([r,alpha])
            full_state[agent_id] = {'obs':np.asarray(obs_dict[agent_id]), 
                                    'state':np.concatenate((obs_dict[agent_id],global_state))}
        return full_state

    def step(self, action_dict):
        obs_dict = {}
        locations = []
        full_state = {}
        reward_dict = {}
        done_dict = {'__all__':False}
        info_dict = {}

        # Targets move (t -> t+1)
        for n in range(self.num_targets):
            self.targets[n].update() 
            locations.append(self.targets[n].state[:2])
        # Agents move (t -> t+1) and observe the targets
        for ii, agent_id in enumerate(action_dict):
            obs_dict[self.agents[ii].agent_id] = []
            reward_dict[self.agents[ii].agent_id] = []
            done_dict[self.agents[ii].agent_id] = []

            action_vw = self.action_map[action_dict[agent_id]]
            _ = self.agents[ii].update(action_vw, [t.state[:2] for t in self.targets])
            locations.append(self.agents[ii].state[:2])
            
            observed = []
            for jj in range(self.num_targets):
                # Observe
                obs = self.observation(self.targets[jj], self.agents[ii])
                observed.append(obs[0])
                self.belief_targets[jj].predict() # Belief state at t+1
                if obs[0]: # if observed, update the target belief.
                    self.belief_targets[jj].update(obs[1], self.agents[ii].state)

            obstacles_pt = map_utils.get_closest_obstacle(self.MAP, self.agents[ii].state)

            if obstacles_pt is None:
                obstacles_pt = (self.sensor_r, np.pi)
            for kk in range(self.num_targets):
                r_b, alpha_b = util.relative_distance_polar(self.belief_targets[kk].state[:2],
                                        xy_base=self.agents[ii].state[:2], 
                                        theta_base=self.agents[ii].state[-1])
                r_dot_b, alpha_dot_b = util.relative_velocity_polar(
                                        self.belief_targets[kk].state[:2],
                                        self.belief_targets[kk].state[2:],
                                        self.agents[ii].state[:2], self.agents[ii].state[-1],
                                        action_vw[0], action_vw[1])
                obs_dict[agent_id].extend([r_b, alpha_b, r_dot_b, alpha_dot_b,
                                        np.log(LA.det(self.belief_targets[kk].cov)), 
                                        float(observed[kk])])
            obs_dict[agent_id].extend([obstacles_pt[0], obstacles_pt[1]])
        #Global state for each agent (ref is origin)
        global_state = util.global_relative_measure(np.array(locations), self.MAP.origin)
        # Full state dict         
        for m, agent_id in enumerate(obs_dict):
            for p, ids in enumerate(obs_dict):
                if agent_id != ids:
                    # Relative location and recent action of all other agents
                    r, alpha = util.relative_distance_polar(np.array(self.agents[p].state[:2]),
                                            xy_base=self.agents[m].state[:2], 
                                            theta_base=self.agents[m].state[2])
                    obs_dict[agent_id].extend([r,alpha])
            full_state[agent_id] = {'obs':np.asarray(obs_dict[agent_id]), 
                                    'state':np.concatenate((obs_dict[agent_id],global_state))}
        # Get all rewards after all agents and targets move (t -> t+1)
        reward, done, mean_nlogdetcov = self.get_reward(obstacles_pt, observed, self.is_training)
        reward_dict['__all__'], done_dict['__all__'], info_dict['mean_nlogdetcov'] = reward, done, mean_nlogdetcov
        return full_state, reward_dict, done_dict, info_dict