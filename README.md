# Target Tracking Environment for Reinforcement Learning (OpenAI gym framework)
This repository contains target tracking environments for reinforcement learning (RL) presented in the following paper which applies RL to active target tracking problems.
* Learning Q-network for Active Information Acquisition (https://arxiv.org/abs/1910.10754) : H. Jeong, B. Schlotfeldt, H. Hassani, M. Morari, D. D. Lee, and G. J. Pappas, “Learning Q-network for Active Information Acquisition,”, IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), Macao, China, 2019

It is built with the OpenAI Gym framework and can be run in the same way as you run an OpenAI gym environment.

## Installation
* Requirements : python3
Install OpenAI gym (http://gym.openai.com/docs/)
```
pip install gym
```
Clone the repository
```
git clone https://github.com/coco66/ttenv.git
```
run_example.py shows a brief example of using the ttenv library with a random action selection.
```
cd ttenv && python run_example.py --render 1
```

## Environments:
See the description in target_tracking.py for more details.
* TargetTracking-v0 : A static target model with Kalman Filter belief tracker.
Targets only move with a small noise in the x and y-axis and Kalman Filter is used for the belief estimate. A target state of each target consists of its x,y position.

* TargetTracking-v1 : A double integrator target model with Kalman Filter belief tracker.
A dynamic model of each target follows a double integrator and Kalman Filter is used for the belief estimate. A target state of each target consists of its x,y position and xdot, ydot.

* TargetTracking-v2 : Predefined target paths with Kalman Filter belief tracker.
It uses a predefined target path and the path file must be provided. Similar to the v1 environment, it uses a Kalman Filter with a double integrator model for the belief estimate.

* TargetTracking-v3 : SE2 Target model with UKF belief tracker
A dynamic model of each target follows the SE2 dynamics with a certain control policy for linear and angular velocities. The target state consists of x, y, and yaw.

* TargetTracking-v4 : SE2 Target model with UKF belief tracker (including velocity estimates).
Similar to the v3 environment but the UKF estimates not only x, y, and yaw but also the control policy (linear and angular velocities) of each target.

### Initialization
Initial positions of targets and the agent significantly influence the reward.  
Currently, you can randomly initialize them within a certain conditions or provide a set of initial position as a pickle file.
* Random initialization  
  The agent's xy position and yaw are always randomly generalized in a given map. If the map is empty, the agent's initial position is the center of the map and only its yaw is randomly initialized. Values for the following variables can be provided (or use the default values) for the initialization of targets and target beliefs.
  * A range for the linear distance from the agent to a target or a belief target.
  * A range for the angular distance from the agent to a target.
  * A range for the angular distance from the agent to a belief target.
  * A boolean variable whether there is an obstacle between the agent and a target.
* Initialization with a given a list of poses
  You can generate a list of initial positions for the agent, targets, and belief targets using ttenv/gen_init_pose.py.

## Running with RL
Examples of learning a deep reinforcement learning policy can be found in the ADFQ repository (https://github.com/coco66/ADFQ).
* DQN : ADFQ/deep_adfq/baselines0/deepq/run_tracking.py
* Double DQN : ADFQ/deep_adfq/baselines0/deepq/run_tracking.py --double_q=1
* Deep ADFQ : ADFQ/deep_adfq/run_tracking.py

## Running Infoplanner (https://bitbucket.org/brentsc/infoplanner.git)
A search-based planning method used as a baseline in the above paper, Learning Q-network for Active Information Acquisition, can be used with this library. To use, download and install the InfoPlanner (https://bitbucket.org/brentsc/infoplanner.git) following its installation instruction. Note that it works only with python version 3.5.
```
export PYTHONPATH="${PYTHONPATH}:YOUR-PATH-TO-INFOPLANNER/lib"
python ttenv/infoplanner_python/run_anytime_planner.py --render 1
```

## Citing
If you use this repo in your research, you can cite it as follows:
```bibtex
@misc{ttenv,
    author = {Heejin Jeong, Brent Schlotfeldt, Hamed Hassani, Manfred Morari, Daniel D. Lee, and George J. Pappas},
    title = {Target tracking environments for Reinforcement Learning},
    year = {2019},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/coco66/ttenv.git}},
}
