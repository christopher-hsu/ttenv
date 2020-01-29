from gym import wrappers
from maTTenv.envs.maTracking_v0 import maTrackingEnv0
from maTTenv.envs.maTracking_v1 import maTrackingEnv1
from maTTenv.envs.maTracking_v2 import maTrackingEnv2

def make(env_name, render=False, figID=0, record=False, ros=False, directory='',
                                        T_steps=None, num_targets=1, **kwargs):
    """
    env_name : str
        name of an environment. (e.g. 'Cartpole-v0')
    type : str
        type of an environment. One of ['atari', 'classic_control',
        'classic_mdp','target_tracking']
    """
    if T_steps is None:
        if num_targets > 1:
            T_steps = 200
        else:
            T_steps = 150
    if env_name == 'maTracking-v0':
        env0 = maTrackingEnv0(**kwargs)
    elif env_name == 'maTracking-v1':
        env0 = maTrackingEnv1(**kwargs)
    elif env_name == 'maTracking-v2':
        env0 = maTrackingEnv2(**kwargs)

    # elif env_name == 'TargetTracking-info1':
    #     from ttenv.infoplanner_python.target_tracking_infoplanner import TargetTrackingInfoPlanner1
    #     env0 = TargetTrackingInfoPlanner1(**kwargs)
    else:
        raise ValueError('No such environment exists.')

    env = wrappers.TimeLimit(env0, max_episode_steps=T_steps)
    if ros:
        from ttenv.ros_wrapper import Ros
        env = Ros(env)
    if render:
        from ttenv.display_wrapper import Display2D
        env = Display2D(env, figID=figID)
    if record:
        from ttenv.display_wrapper import Video2D
        env = Video2D(env, dirname = directory)

    return env