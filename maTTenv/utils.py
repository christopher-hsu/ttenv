import numpy as np 
def wrap_around(x):
    # x \in [-pi,pi)
    if x >= np.pi:
        return x - 2*np.pi
    elif x < -np.pi:
        return x + 2*np.pi
    else:
        return x

def vectorized_wrap_around(x):
    x[x>=np.pi] -= 2*np.pi
    x[x<-np.pi] += 2*np.pi
    return x

def relative_measure(x_target, x_main):
    diff = x_target[:2] - x_main[:2]
    r = np.sqrt(np.sum(diff**2))
    alpha = wrap_around(np.arctan2(diff[1],diff[0]) - x_main[2])
    return r, alpha, diff

def global_relative_measure(x_target, x_main):
    diff = x_target - x_main
    r = np.sqrt(np.sum(diff**2, axis=1))
    alpha = vectorized_wrap_around(np.arctan2(diff[:,1],diff[:,0]))
    #Rearrange to [r,alpha]*(nb_agents,nb_targets)
    global_state = np.ndarray.flatten(np.vstack((r,alpha)).T)
    return global_state

# def coord_change2b(vec, ang):
#     assert(len(vec) == 2)
#     # R^T * v
#     return np.matmul([[np.cos(ang), np.sin(ang)], [-np.sin(ang), np.cos(ang)]], vec)

def xyg2polarb(xy_target, xy_b, theta_b):
    # Transformation of xy_target (x,y) in the global frame to polar
    # coordinate with respect to the base frame located at xy_b and rotated
    # by theta.
    diff = xy_target - xy_b
    r_target_b = np.sqrt(np.sum(diff**2))
    theta_target_b = wrap_around(np.arctan2(diff[1],diff[0]) - theta_b)
    return r_target_b, theta_target_b, diff

def xyg2polarb_dot(xyth_target, vw_target, xyth_b, vw_b):
    # Radial and angular velocity of the target with respect to the base frame
    # located at xy_b with a rotation of theta_b, moving with a linear
    # velocity v and angular velocity w. This function is designed specifically
    # for the SE2 Agent and SE2 Target target case.
    r_target_b = np.sqrt(np.sum((xyth_target[:2] - xyth_b[:2])**2))
    if r_target_b == 0.0:
        return 0.0, 0.0
    xy_dot_b = vw_to_xydot(vw_b[0], vw_b[1], np.sqrt(np.sum(xyth_b[:2]**2)), xyth_b[2])
    xy_dot_target = vw_to_xydot(vw_target[0], vw_target[1],
                            np.sqrt(np.sum(xyth_target[:2]**2)), xyth_target[2])
    r_dot_target_b = np.sum((xyth_target[:2] - xyth_b[:2]) \
                        * (np.array(xy_dot_target) - np.array(xy_dot_b))) / r_target_b
    theta_dot_target_b = vw_target[1] - vw_b[1]
    return r_dot_target_b, theta_dot_target_b

def xyg2polarb_dot_2(xy_target, xy_dot_target, xy_b, theta_b, v_b, w_b):
    # Radial and angular velocity of the target with respect to the base frame
    # located at xy_b with a rotation of theta_b, moving with a linear
    # velocity v and angular velocity w. This function is designed specifically
    # for the SE2 Agent and Double Integrator target case.
    r_target_b = np.sqrt(np.sum((xy_target - xy_b)**2))
    if r_target_b == 0.0:
        return 0.0, 0.0
    xy_dot_b = vw_to_xydot(v_b, w_b, np.sqrt(np.sum(xy_b**2)), theta_b)
    r_dot_target_b = np.sum((xy_target - xy_b) \
                        * (xy_dot_target - np.array(xy_dot_b))) / r_target_b
    _, theta_dot_target = xydot_to_vw(xy_target[0], xy_target[1],
                                            xy_dot_target[0], xy_dot_target[1])
    theta_dot_target_b = theta_dot_target - w_b
    return r_dot_target_b, theta_dot_target_b

def vw_to_xydot(v, w, r, theta):
    return v*np.cos(theta)-r*w*np.sin(theta), v*np.sin(theta)+r*w*np.cos(theta)

def xydot_to_vw(x, y, x_dot, y_dot):
    r2 = x*x + y*y
    if r2 == 0.0:
        return 0.0, 0.0
    r_dot = (x*x_dot + y*y_dot)/np.sqrt(r2)
    theta_dot = (x*y_dot - x_dot*y)/r2
    return r_dot, theta_dot

def transform_2d(vec, ang, frame_xy = [0.0, 0.0]):
    # Both vec and frame_xy are in the global coordinate. vec is a vector
    # you want to transform with respect to a certain frame which is located at
    # frame_xy with ang.
    # R^T * (vec - frame_xy).
    # R is a rotation matrix of the frame w.r.t the global frame.
    assert(len(vec) == 2)

    return np.matmul([[np.cos(ang), np.sin(ang)], [-np.sin(ang), np.cos(ang)]],
        vec - np.array(frame_xy))
