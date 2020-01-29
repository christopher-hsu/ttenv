import os, copy, pdb
import numpy as np
from numpy import linalg as LA

from gym import spaces, logger

from envs.maps import map_utils
import envs.env_utils as util 
from envs.maTracking.agent_models import *
from envs.maTracking.belief_tracker import KFbelief
from envs.maTracking.metadata import *
from envs.maTracking.maTracking_Base import maTrackingBase

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
                [d, alpha, action] * nb_agents-1
    Agent : SE2 model, [x,y,theta]
    Target : Double Integrator model, [x,y,xdot,ydot]
    Belief Target : KF, Double Integrator model

global state: [d, alpha] * (nb_targets+nb_agents) in ref to origin
"""

class maTrackingEnv2(maTrackingBase):

    def __init__(self, hyp, q_true = 0.02, q = 0.01, is_training=True, known_noise=True):
        super().__init__(hyp, agent_dim=3, target_dim=4, q_true=q_true, q=q)

        self.target_init_vel = TARGET_INIT_VEL*np.ones((2,))
        # LIMIT
        self.vel_limit = np.array([2.0, 2.0])
        self.limit = {} # 0: low, 1:highs
        self.limit['agent'] = [np.concatenate((self.MAP.mapmin,[-np.pi])), np.concatenate((self.MAP.mapmax, [np.pi]))]
        self.limit['target'] = [np.concatenate((self.MAP.mapmin,  -self.vel_limit)), np.concatenate((self.MAP.mapmax, self.vel_limit))]
        self.limit['state'] = [np.concatenate(([0.0, -np.pi, -2*self.vel_limit[0], -2*self.vel_limit[1], -50.0, 0.0]*self.num_targets, 
                                np.concatenate(([0.0, -np.pi ],
                                [0.0, -np.pi, 0.0]*(self.num_agents-1))))),
                                np.concatenate(([600.0, np.pi, 2*self.vel_limit[0], 2*self.vel_limit[1],  50.0, 2.0]*self.num_targets, 
                                np.concatenate(([self.sensor_r, np.pi],
                                [600.0, np.pi, 12.0]*(self.num_agents-1)))))]
        self.limit['global'] = [np.concatenate((self.limit['state'][0],np.array([0.0,-np.pi]*(self.num_targets+self.num_agents)))),
                                np.concatenate((self.limit['state'][1],np.array([600.0,np.pi]*(self.num_targets+self.num_agents))))]
        self.observation_space = spaces.Box(self.limit['state'][0], self.limit['state'][1], dtype=np.float32)
        self.global_space = spaces.Box(self.limit['global'][0], self.limit['global'][1], dtype=np.float32)
        self.targetA = np.concatenate((np.concatenate((np.eye(2), self.sampling_period*np.eye(2)), axis=1), 
                                        [[0,0,1,0],[0,0,0,1]]))
        self.target_noise_cov = self.q * np.concatenate((
                        np.concatenate((self.sampling_period**3/3*np.eye(2), self.sampling_period**2/2*np.eye(2)), axis=1),
                        np.concatenate((self.sampling_period**2/2*np.eye(2), self.sampling_period*np.eye(2)),axis=1) ))
        if known_noise:
            self.target_true_noise_sd = self.target_noise_cov
        else:
            self.target_true_noise_sd = q_true * np.concatenate((
                        np.concatenate((self.sampling_period**2/2*np.eye(2), self.sampling_period/2*np.eye(2)), axis=1),
                        np.concatenate((self.sampling_period/2*np.eye(2), self.sampling_period*np.eye(2)),axis=1) ))

        # Build a robot 
        self.setup_agents()
        # Build a target
        self.setup_targets()
        self.setup_belief_targets()

    def setup_agents(self):
        self.agents = [AgentSE2(agent_id = 'agent-' + str(i), 
                        dim=self.agent_dim, sampling_period=self.sampling_period, limit=self.limit['agent'], 
                        collision_func=lambda x: map_utils.is_collision(self.MAP, x, MARGIN2WALL), 
                        margin=MARGIN)
                        for i in range(self.num_agents)]

    def setup_targets(self):
        self.targets = [AgentDoubleInt2D(agent_id = 'agent-' + str(i),
                        dim=self.target_dim, sampling_period=self.sampling_period, limit=self.limit['target'],
                        collision_func=lambda x: map_utils.is_collision(self.MAP, x, MARGIN2WALL),
                        margin=MARGIN, A=self.targetA, W=self.target_true_noise_sd) 
                        for i in range(self.num_targets)]

    def setup_belief_targets(self):
        self.belief_targets = [KFbelief(agent_id = 'agent-' + str(i),
                        dim=self.target_dim, limit=self.limit['target'], A=self.targetA,
                        W=self.target_noise_cov, obs_noise_func=self.observation_noise, 
                        collision_func=lambda x: map_utils.is_collision(self.MAP, x, MARGIN2WALL))
                        for i in range(self.num_targets)]

    def reset(self,init_random=True):
        """
        Agents are given random positions in the map, targets are given random positions near a random agent.
        Return a full state dict with agent ids (keys) that refer to their observation and global state
        """
        obs_state = {}
        locations = []
        global_state = {}
        full_state = {}
        if init_random:
            agents_init = np.zeros((self.num_agents,2))
            for ii in range(self.num_agents):
                isvalid = False 
                while(not isvalid):
                    a_init = np.random.random((2,))*(self.MAP.mapmax-self.MAP.mapmin) + self.MAP.mapmin     
                    isvalid = not(map_utils.is_collision(self.MAP, a_init, MARGIN2WALL))
                agents_init[ii,:] = a_init
                self.agents[ii].reset([a_init[0], a_init[1], np.random.random()*2*np.pi-np.pi])
                obs_state[self.agents[ii].agent_id] = []

            for jj in range(self.num_targets):
                isvalid = False
                while(not isvalid):            
                    rand_ang = np.random.rand()*2*np.pi - np.pi 
                    t_r = INIT_DISTANCE
                    rand_a = np.random.randint(self.num_agents)
                    rand_a_init = agents_init[rand_a,:]
                    t_init = np.array([t_r*np.cos(rand_ang), t_r*np.sin(rand_ang)]) + rand_a_init
                    if (np.sqrt(np.sum((t_init - rand_a_init)**2)) < MARGIN):
                        isvalid = False
                    else:
                        isvalid = not(map_utils.is_collision(self.MAP, t_init, MARGIN2WALL))
                self.belief_targets[jj].reset(init_state=np.concatenate((t_init + 10*(np.random.rand(2)-0.5), np.zeros(2))),
                                init_cov=self.target_init_cov)
                self.targets[jj].reset(np.concatenate((t_init, self.target_init_vel)))
                locations.append(self.targets[jj].state[:2])
                for kk in range(self.num_agents):
                    r, alpha, _ = util.relative_measure(self.belief_targets[jj].state, self.agents[kk].state)
                    logdetcov = np.log(LA.det(self.belief_targets[jj].cov))
                    obs_state[self.agents[kk].agent_id].extend([r, alpha, 0.0, 0.0, logdetcov, 0.0])
        # All targets and agents locations relative to map origin (targets then agents)
        for n in range(self.num_agents):
            locations.append(self.agents[n].state[:2])
        global_state = util.global_relative_measure(np.array(locations), self.MAP.origin)
        # Full state dict         
        for m, agent_id in enumerate(obs_state):
            obs_state[agent_id].extend([self.sensor_r, np.pi])
            # Relative location and past action of all other agents
            for p, ids in enumerate(obs_state):
                if agent_id != ids:
                    r, alpha, _ = util.relative_measure(np.array(self.agents[p].state), self.agents[m].state)
                    obs_state[agent_id].extend([r,alpha,0.0])
            full_state[agent_id] = {'obs':np.asarray(obs_state[agent_id]), 
                                    'state':np.concatenate((obs_state[agent_id],global_state))}
        return full_state

    def step(self, action_dict):
        obs_state = {}
        locations = []
        full_state = {}
        reward_dict = {}
        done_dict = {'__all__':False}
        info_dict = {}

        for n in range(self.num_targets):
            self.targets[n].update() 
            locations.append(self.targets[n].state[:2])
        tot_observed = []
        for ii, agent_id in enumerate(action_dict):
            obs_state[self.agents[ii].agent_id] = []
            reward_dict[self.agents[ii].agent_id] = []
            done_dict[self.agents[ii].agent_id] = []

            action_val = self.action_map[action_dict[agent_id]]
            boundary_penalty = self.agents[ii].update(action_val, [t.state[:2] for t in self.targets])
            obstacles_pt = map_utils.get_cloest_obstacle(self.MAP, self.agents[ii].state)
            locations.append(self.agents[ii].state[:2])
            
            observed = []
            for jj in range(self.num_targets):    
                # Observe
                obs = self.observation(self.targets[jj], self.agents[ii])
                observed.append(obs[0])
                tot_observed.append(obs[0])
                # Update the belief of the agent on the target using KF
                self.belief_targets[jj].update(obs[0], obs[1], self.agents[ii].state)
            reward, done, test_reward = self.get_reward(obstacles_pt, observed, tot_observed, self.is_training)
            reward_dict[agent_id] = reward
            done_dict[agent_id] = done
            info_dict[agent_id] = test_reward

            if obstacles_pt is None:
                obstacles_pt = (self.sensor_r, np.pi)
            for kk in range(self.num_targets):
                r_b, alpha_b, _ = util.relative_measure(self.belief_targets[kk].state, self.agents[ii].state)
                rel_target_vel = util.coord_change2b(self.belief_targets[kk].state[2:], alpha_b+self.agents[ii].state[-1])
                obs_state[agent_id].extend([r_b, alpha_b, 
                                            rel_target_vel[0], rel_target_vel[1],
                                            np.log(LA.det(self.belief_targets[kk].cov)), float(observed[kk])])
            obs_state[agent_id].extend([obstacles_pt[0], obstacles_pt[1]])
        #Global state for each agent (ref is origin)
        global_state = util.global_relative_measure(np.array(locations), self.MAP.origin)
        # Full state dict         
        for m, agent_id in enumerate(obs_state):
            for p, ids in enumerate(obs_state):
                if agent_id != ids:
                    # Relative location and recent action of all other agents
                    r, alpha, _ = util.relative_measure(np.array(self.agents[p].state), self.agents[m].state)
                    obs_state[agent_id].extend([r,alpha,action_dict[ids]])
            full_state[agent_id] = {'obs':np.asarray(obs_state[agent_id]), 
                                    'state':np.concatenate((obs_state[agent_id],global_state))}
        return full_state, reward_dict, done_dict, info_dict