import math
import numpy as np
import transforms3d
from environments.urban_environment import carla_config

def pedestrian_relative_position(ped_trans, ev_trans):

    p_xyz = np.array([ped_trans.location.x, ped_trans.location.y, ped_trans.location.z])
    ev_xyz =  np.array([ev_trans.location.x, ev_trans.location.y, ev_trans.location.z])
    ped_loc = p_xyz - ev_xyz

    pitch = math.radians(ev_trans.rotation.pitch)
    roll = math.radians(ev_trans.rotation.roll)
    yaw = math.radians(ev_trans.rotation.yaw)
    R = transforms3d.euler.euler2mat(roll, pitch, yaw).T
    ped_loc_relative = np.dot(R, ped_loc)

    return ped_loc_relative
