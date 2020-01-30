import os, copy, pdb
import numpy as np
from numpy import linalg as LA
from gym import spaces, logger
from maTTenv.maps import map_utils
import maTTenv.utils as util 
from maTTenv.agent_models import *
from maTTenv.belief_tracker import KFbelief
from maTTenv.metadata import METADATA_v0 as METADATA
from maTTenv.envs.maTracking_Base import maTrackingBase

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
    RL state: [d, alpha, logdet(Sigma)] * nb_targets , [o_d, o_alpha]
    Agent:
    Target: Static [x,y] + noise
    Belief Target: KF, Estimate only x and y
"""

class maTrackingEnv0(maTrackingBase):

    def __init__(self, num_agents=2, num_targets=1, map_name='empty', is_training=True, known_noise=True, **kwargs):
        super().__init__(num_agents=num_agents, num_targets=num_targets, map_name=map_name)
       
        self.id = 'maTracking-v0'
        self.agent_dim = 3
        self.target_dim = 2
        self.limit['global'] = np.array([[0.0,-np.pi]*(self.num_targets+self.num_agents),
                                [600.0,np.pi]*(self.num_targets+self.num_agents)])
        self.global_space = spaces.Box(self.limit['global'][0], self.limit['global'][1], dtype=np.float32)
       
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

    def reset(self, init_random=True):
    # def reset(self, **kwargs):
        """
        Agents are given random positions in the map, targets are given random positions near a random agent.
        Return a RL state dict with agent ids (keys) and target state info
        """
        obs_state = {}
        locations = []
        global_state = {}
        full_state = {}
        # init_pose = self.get_init_pose(**kwargs)

        if init_random:
            agents_init = np.zeros((self.num_agents,2))
            # initialize agent locations randomly in map
            for ii in range(self.num_agents):
                isvalid = False 
                while(not isvalid):
                    a_init = np.random.random((2,))*(self.MAP.mapmax-self.MAP.mapmin) + self.MAP.mapmin     
                    isvalid = not(map_utils.is_collision(self.MAP, a_init))
                agents_init[ii,:] = a_init
                self.agents[ii].reset([a_init[0], a_init[1], np.random.random()*2*np.pi-np.pi])
                obs_state[self.agents[ii].agent_id] = []
            # initialize targets near random agent
            for jj in range(self.num_targets):
                isvalid = False
                while(not isvalid):            
                    rand_ang = np.random.rand()*2*np.pi - np.pi 
                    t_r = METADATA['init_distance']
                    rand_a = np.random.randint(self.num_agents)
                    rand_a_init = agents_init[rand_a,:]
                    t_init = np.array([t_r*np.cos(rand_ang), t_r*np.sin(rand_ang)]) + rand_a_init
                    if (np.sqrt(np.sum((t_init - rand_a_init)**2)) < METADATA['margin']):
                        isvalid = False
                    else:
                        isvalid = not(map_utils.is_collision(self.MAP, t_init))
                self.belief_targets[jj].reset(init_state=t_init + 10*(np.random.rand(2)-0.5),
                                init_cov=self.target_init_cov)
                self.targets[jj].reset(t_init)
                locations.append(self.targets[jj].state[:2])
                #For each agent calculate belief of all targets
                for kk in range(self.num_agents):
                    r, alpha, _ = util.relative_measure(self.belief_targets[jj].state, self.agents[kk].state)
                    logdetcov = np.log(LA.det(self.belief_targets[jj].cov))
                    obs_state[self.agents[kk].agent_id].extend([r, alpha, logdetcov, 0.0])
        # All targets and agents locations relative to map origin (targets then agents)
        for n in range(self.num_agents):
            locations.append(self.agents[n].state[:2])
        global_state = util.global_relative_measure(np.array(locations), self.MAP.origin)
        # Full state dict         
        for agent_id in obs_state:
            obs_state[agent_id].extend([self.sensor_r, np.pi])
            full_state[agent_id] = {'obs':np.array(obs_state[agent_id]), 'state':global_state}
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
        for ii, agent_id in enumerate(action_dict):
            obs_state[self.agents[ii].agent_id] = []
            reward_dict[self.agents[ii].agent_id] = []
            done_dict[self.agents[ii].agent_id] = []

            action_val = self.action_map[action_dict[agent_id]]
            boundary_penalty = self.agents[ii].update(action_val, [t.state[:2] for t in self.targets])
            obstacles_pt = map_utils.get_closest_obstacle(self.MAP, self.agents[ii].state)
            locations.append(self.agents[ii].state[:2])
            
            observed = []
            for jj in range(self.num_targets):
                # Observe
                obs = self.observation(self.targets[jj], self.agents[ii])
                observed.append(obs[0])
                # Update the belief of the agent on the target using KF
                self.belief_targets[jj].update(obs[0], obs[1], self.agents[ii].state)
            reward, done, mean_nlogdetcov = self.get_reward(obstacles_pt, observed, self.is_training)
            reward_dict[agent_id] = reward
            done_dict[agent_id] = done
            info_dict[agent_id] = mean_nlogdetcov

            if obstacles_pt is None:
                obstacles_pt = (self.sensor_r, np.pi)
            for kk in range(self.num_targets):
                r_b, alpha_b, _ = util.relative_measure(self.belief_targets[kk].state, self.agents[ii].state)
                obs_state[agent_id].extend([r_b, alpha_b, 
                                        np.log(LA.det(self.belief_targets[kk].cov)), float(observed[kk])])
            obs_state[agent_id].extend([obstacles_pt[0], obstacles_pt[1]])
            full_state[agent_id] = {'obs':np.array(obs_state[agent_id]), 'state':[]}
        #Global state for each agent (ref is origin)
        global_state = util.global_relative_measure(np.array(locations), self.MAP.origin)
        for agent_id in full_state:
            full_state[agent_id]['state'] = global_state
        return full_state, reward_dict, done_dict, info_dict