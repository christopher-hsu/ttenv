"""Dynamic Object Models

AgentDoubleInt2D : Double Integrator Model in 2D
                   state: x,y,xdot,ydot
AgentSE2 : SE2 Model 
           state x,y,theta

Agent2DFixedPath : Model with a pre-defined path
Agent_InfoPlanner : Model from the InfoPlanner repository

SE2Dynamics : update dynamics function with a control input -- linear, angular velocities
SEDynamicsVel : update dynamics function for contant linear and angular velocities

edited by christopher-hsu from coco66 for multi_agent
"""

import numpy as np
from envs.maTracking.metadata import *
import envs.env_utils as util
# import pyInfoGathering as IGL

class Agent(object):
    def __init__(self, agent_id, dim, sampling_period, limit, collision_func, margin):
        self.agent_id = agent_id
        self.dim = dim
        self.sampling_period = sampling_period
        self.limit = limit
        self.collision_func = collision_func
        self.margin = margin

    def range_check(self):
        self.state = np.clip(self.state, self.limit[0], self.limit[1])

    def collision_check(self, pos):
        return self.collision_func(pos[:2])

    def margin_check(self, pos, target_pos):
        return np.sqrt(np.sum((pos - target_pos)**2)) < self.margin # no update 

    def reset(self, init_state):
        self.state = init_state


class AgentDoubleInt2D(Agent):
    def __init__(self, agent_id, dim, sampling_period, limit, collision_func, 
                    margin, A=None, W=None):
        super().__init__(agent_id, dim, sampling_period, limit, collision_func, margin)
        self.A = np.eye(self.dim) if A is None else A
        self.W = W

    def update(self, margin_pos=None):
        new_state = np.matmul(self.A, self.state)
        if self.W is not None:
            noise_sample = np.random.multivariate_normal(np.zeros(self.dim,), self.W)
            new_state += noise_sample
        if self.collision_check(new_state[:2]):
            new_state = self.collision_control(new_state)
        self.state = new_state
        self.range_check()

    def collision_control(self, new_state):
        new_state[0] = self.state[0]
        new_state[1] = self.state[1]
        if self.dim > 2:
            new_state[2] = -2 * .2 * new_state[2] + np.random.normal(0.0, 0.2)#-0.001*np.sign(new_state[2])
            new_state[3] = -2 * .2 * new_state[3] + np.random.normal(0.0, 0.2)#-0.001*np.sign(new_state[3])
        return new_state

class AgentSE2(Agent):
    def __init__(self, agent_id, dim, sampling_period, limit, collision_func, margin, policy=None):
        super().__init__(agent_id, dim, sampling_period, limit, collision_func, margin)
        self.policy = policy

    def update(self, control_input=None, margin_pos=None, col=False):
        """
        control_input : [linear_velocity, angular_velocity]
        margin_pos : a minimum distance to a target
        """
        if control_input is None:
            control_input = self.policy.get_control(self.state)

        new_state = SE2Dynamics(self.state, self.sampling_period, control_input) 
        is_col = 0 
        if self.collision_check(new_state[:2]):
            is_col = 1
            new_state[:2] = self.state[:2]
            if self.policy is not None:
                self.policy.collision(new_state)

        elif margin_pos is not None:
            if type(margin_pos) != list:
                margin_pos = [margin_pos]
            for mp in margin_pos:
                if self.margin_check(new_state[:2], margin_pos):
                    new_state[:2] = self.state[:2]
                    break        
        self.state = new_state
        self.range_check()        

        return is_col

def SE2Dynamics(x, dt, u):  
    assert(len(x)==3)          
    tw = dt * u[1] 
    # Update the agent state 
    if abs(tw) < 0.001:
        diff = np.array([dt*u[0]*np.cos(x[2]+tw/2),
                dt*u[0]*np.sin(x[2]+tw/2),
                tw])
    else:
        diff = np.array([u[0]/u[1]*(np.sin(x[2]+tw) - np.sin(x[2])),
                u[0]/u[1]*(np.cos(x[2]) - np.cos(x[2]+tw)),
                tw])
    new_x = x + diff
    new_x[2] = util.wrap_around(new_x[2])
    return new_x

def SE2DynamicsVel(x, dt, u=None):  
    assert(len(x)==5) # x = [x,y,theta,v,w]
    odom = SE2Dynamics(x[:3], dt, x[-2:])
    return np.concatenate((odom, x[-2:]))

class Agent2DFixedPath(Agent):
    def __init__(self, dim, sampling_period, limit, collision_func, path, margin):
        Agent.__init__(self, dim, sampling_period, limit, collision_func, margin)
        self.path = path

    def update(self, margin_pos=None):
        # fixed policy for now
        self.t += 1
        self.state = np.concatenate((self.path[self.t][:2], self.path[self.t][-2:]))

    def reset(self, init_state):
        self.t = 0
        self.state = np.concatenate((self.path[self.t][:2], self.path[self.t][-2:]))
        


class Agent_InfoPlanner(Agent):
    def __init__(self,  dim, sampling_period, limit, collision_func,  
                    se2_env, sensor_obj, margin):
        Agent.__init__(self, dim, sampling_period, limit, collision_func, margin)
        self.se2_env = se2_env
        self.sensor = sensor_obj
        self.sampling_period = sampling_period
        self.action_map = {}        
        for (i,v) in enumerate([3,2,1,0]):
            for (j,w) in enumerate([np.pi/2, 0, -np.pi/2]):
                self.action_map[3*i+j] = (v,w)

    def reset(self, init_state, belief_target):
        self.agent = IGL.Robot(init_state, self.se2_env, belief_target, self.sensor)
        self.state = self.get_state()
        return self.state

    def update(self, action, target_state):
        action =  self.update_filter(action, target_state)
        self.agent.applyControl([int(action)], 1)
        self.state = self.get_state()

    def get_state(self):
        return np.concatenate((self.agent.getState().position[:2], [self.agent.getState().getYaw()]))

    def get_state_object(self):
        return self.agent.getState()

    def observation(self, target_obj):
        return self.agent.sensor.senseMultiple(self.get_state_object(), target_obj)

    def get_belief_state(self):
        return self.agent.tmm.getTargetState()

    def get_belief_cov(self):
        return self.agent.tmm.getCovarianceMatrix()

    def update_belief(self, GaussianBelief):
        self.agent.tmm.updateBelief(GaussianBelief.mean, GaussianBelief.cov)

    def update_filter(self, action, target_state):
        state = self.get_state()
        control_input = self.action_map[action]
        tw = self.sampling_period*control_input[1]
        # Update the agent state 
        if abs(tw) < 0.001:
            diff = np.array([self.sampling_period*control_input[0]*np.cos(state[2]+tw/2),
                    self.sampling_period*control_input[0]*np.sin(state[2]+tw/2),
                    tw])
        else:
            diff = np.array([control_input[0]/control_input[1]*(np.sin(state[2]+tw) - np.sin(state[2])),
                    control_input[0]/control_input[1]*(np.cos(state[2]) - np.cos(state[2]+tw)),
                    tw])
        new_state = state + diff
        if len(target_state.shape)==1:
            target_state = [target_state]
        target_col = False
        for n in range(target_state.shape[0]): # For each target
            target_col = np.sqrt(np.sum((new_state[:2] - target_state[n][:2])**2)) < margin
            if target_col:
                break

        if self.collision_check(new_state) or target_col: # no update 
            new_action = 9 + action%3
        else:
            new_action = action
        return new_action






