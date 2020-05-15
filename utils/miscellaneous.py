import math
import numpy as np
import transforms3d
from environments.urban_environment import carla_config
import numpy as np
import matplotlib.pyplot as plt

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



def plot_grid(data, x_range = 10, y_range = 10, title = 'None'):

    fig, ax = plt.subplots()
    ax.set_ylim(y_range - 0.5, -0.5)
    ax.set_title(title)
    fig.subplots_adjust(bottom = 0.15, left = 0.2)

    color_map = plt.cm.get_cmap('Blues_r')
    #reversed_color_map = color_map.reversed()

    ax.matshow(data, cmap=color_map)

    print('This: ', np.where(data == 1.0))

    data[0][9] = 1.0

    for row in range(x_range):
        for col in range(y_range):
            c = data[col, row]
            if c == 1:
                print(row, col)

            c = round(c,2)
            ax.text(row, col, str(c), va='center', ha='center')

    plt.gca().invert_yaxis()
    plt.show()
    input('Enter to close: ')
    plt.close()


# intersection_matrix = np.zeros([32, 32])
# intersection_matrix[10][10] = 1.0

# plot_grid(data = intersection_matrix, x_range = 32, y_range = 32, title = 'Position matrix')
