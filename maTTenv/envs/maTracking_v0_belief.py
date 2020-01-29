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

    def __init__(self, hyp, q_true = 0.01, q = 0.01, is_training=True, known_noise=True):
        super().__init__(hyp, agent_dim=3, target_dim=2, q_true=q_true, q=q)

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
        self.targets = [AgentDoubleInt2D(agent_id = 'target-' + str(i),
                        dim=self.target_dim, sampling_period=self.sampling_period, limit=self.limit['target'],
                        collision_func=lambda x: map_utils.is_collision(self.MAP, x, MARGIN2WALL),
                        margin=MARGIN, A=self.targetA, W=self.target_true_noise_sd) 
                        for i in range(self.num_targets)]

    def setup_belief_targets(self):
        #keys of dict are agent_ids with values being a list of target beliefs
        self.belief_targets = {}
        for i in range(self.num_agents):
            self.belief_targets[self.agents[i].agent_id] = []
        for agent_id in self.belief_targets:
            self.belief_targets[agent_id] = [KFbelief(agent_id = 'target-' + str(j),
                        dim=self.target_dim, limit=self.limit['target'], A=self.targetA,
                        W=self.target_noise_cov, obs_noise_func=self.observation_noise, 
                        collision_func=lambda x: map_utils.is_collision(self.MAP, x, MARGIN2WALL))
                        for j in range(self.num_targets)]

        # self.belief_targets = [KFbelief(agent_id = 'agent-' + str(i),
        #                 dim=self.target_dim, limit=self.limit['target'], A=self.targetA,
        #                 W=self.target_noise_cov, obs_noise_func=self.observation_noise, 
        #                 collision_func=lambda x: map_utils.is_collision(self.MAP, x, MARGIN2WALL))
        #                 for i in range(self.num_targets)]

    def reset(self,init_random=True):
        """
        Agents are given random positions in the map, targets are given random positions near a random agent.
        Return a RL state dict with agent ids (keys) and target state info
        """
        self.obs_state = {}
        if init_random:
            agents_init = np.zeros((self.num_agents,2))
            # initialize agent locations randomly in map
            for ii in range(self.num_agents):
                isvalid = False 
                while(not isvalid):
                    a_init = np.random.random((2,))*(self.MAP.mapmax-self.MAP.mapmin) + self.MAP.mapmin     
                    isvalid = not(map_utils.is_collision(self.MAP, a_init, MARGIN2WALL))
                agents_init[ii,:] = a_init
                self.agents[ii].reset([a_init[0], a_init[1], np.random.random()*2*np.pi-np.pi])
                self.obs_state[self.agents[ii].agent_id] = []
            # initialize targets near random agent
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
                self.targets[jj].reset(t_init)
                #For each agent calculate belief of all targets
                for kk, agent_id in enumerate(sorted(self.belief_targets)):
                    self.belief_targets[agent_id][jj].reset(init_state=t_init + 10*(np.random.rand(2)-0.5),
                                init_cov=self.target_init_cov)
                    r, alpha, _ = util.relative_measure(self.belief_targets[agent_id][jj].state, self.agents[kk].state)
                    logdetcov = np.log(LA.det(self.belief_targets[agent_id][jj].cov))
                    self.obs_state[agent_id].extend([r, alpha, logdetcov, 0.0])
        for agent_id in self.obs_state:
            self.obs_state[agent_id].extend([self.sensor_r, np.pi])
        return self.obs_state

    def step(self, action_dict):
        self.obs_state = {}
        reward_dict = {}
        # done_dict = {}
        done_dict = {'__all__':False}
        info_dict = {}
        for ii, agent_id in enumerate(action_dict):
            self.obs_state[self.agents[ii].agent_id] = []
            reward_dict[self.agents[ii].agent_id] = []
            done_dict[self.agents[ii].agent_id] = []

            action_val = self.action_map[action_dict[agent_id]]
            boundary_penalty = self.agents[ii].update(action_val, [t.state[:2] for t in self.targets])
            obstacles_pt = map_utils.get_cloest_obstacle(self.MAP, self.agents[ii].state)
            
            observed = []
            for jj in range(self.num_targets):
                self.targets[jj].update(self.agents[ii].state[:2])      #currently does NOT use agent state to update
                # Observe
                obs = self.observation(self.targets[jj], self.agents[ii])
                observed.append(obs[0])
                # Update the belief of the agent on the target using KF
                self.belief_targets[jj].update(obs[0], obs[1], self.agents[ii].state)
            reward, done, test_reward = self.get_reward(obstacles_pt, observed, self.is_training)
            reward_dict[agent_id] = reward
            done_dict[agent_id] = done
            info_dict[agent_id] = test_reward

            if obstacles_pt is None:
                obstacles_pt = (self.sensor_r, np.pi)
            for kk in range(self.num_targets):
                r_b, alpha_b, _ = util.relative_measure(self.belief_targets[kk].state, self.agents[ii].state)
                self.obs_state[agent_id].extend([r_b, alpha_b, 
                                        np.log(LA.det(self.belief_targets[kk].cov)), float(observed[kk])])
            self.obs_state[agent_id].extend([obstacles_pt[0], obstacles_pt[1]])
            # self.state = np.array(self.state)
        return self.obs_state, reward_dict, done_dict, info_dict