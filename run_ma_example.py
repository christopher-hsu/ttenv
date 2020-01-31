import maTTenv
import numpy as np
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--env', help='environment ID', type=str, default='maTracking-v1')
parser.add_argument('--render', help='whether to render', type=int, default=0)
parser.add_argument('--record', help='whether to record', type=int, default=0)
parser.add_argument('--ros', help='whether to use ROS', type=int, default=0)
parser.add_argument('--nb_agents', help='the number of agents', type=int, default=2)
parser.add_argument('--nb_targets', help='the number of targets', type=int, default=1)
parser.add_argument('--log_dir', help='a path to a directory to log your data', type=str, default='.')
parser.add_argument('--map', type=str, default="emptySmall")

args = parser.parse_args()

def main():
    env = maTTenv.make(args.env,
                    render=args.render,
                    record=args.record,
                    ros=args.ros,
                    directory=args.log_dir,
                    map_name=args.map,
                    num_agents=args.nb_agents,
                    num_targets=args.nb_targets,
                    is_training=False,
                    )
    nlogdetcov = []
    action_dict = {}

    obs, done = env.reset(), False
    while(not done):
        if args.render:
            env.render()

        for agent_id, a_obs in obs.items():
            action_dict[agent_id] = env.action_space.sample()

        obs, rew, done, info = env.step(action_dict)
        done = done['__all__']
        # nlogdetcov.append(info['mean_nlogdetcov'])

    print("Sum of negative logdet of the target belief covariances : %.2f"%np.sum(nlogdetcov))

if __name__ == "__main__":
    main()
    """
    Examples:
        >>> env = MyMultiAgentEnv()
        >>> obs = env.reset()
        >>> print(obs)
        {
            "agent_0": [2.4, 1.6],
            "car_1": [3.4, -3.2],
            "traffic_light_1": [0, 3, 5, 1],
        }
        >>> obs, rewards, dones, infos = env.step(
            action_dict={
                "agent_0": 1, "car_1": 0, "traffic_light_1": 2,
            })
        >>> print(rewards)
        {
            "agent_0": 3,
            "car_1": -1,
            "traffic_light_1": 0,
        }
        >>> print(dones)
        {
            "agent_0": False,  # agent_0 is still running
            "car_1": True,     # car_1 is done
            "__all__": False,  # the env is not done
        }
        >>> print(infos)
        {
            "agent_0": {},  # info for agent_0
            "car_1": {},    # info for car_1
        }
    """