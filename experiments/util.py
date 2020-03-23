
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch

def get_output_folder(parent_dir, env_name):
    """Return save folder.
    Assumes folders in the parent_dir have suffix -run{run
    number}. Finds the highest run number and sets the output folder
    to that number + 1. This is just convenient so that if you run the
    same script multiple times tensorboard can plot all of the results
    on the same plots with different names.
    Parameters
    ----------
    parent_dir: str
      Path of the directory containing all experiment runs.
    Returns
    -------
    parent_dir/run_dir
      Path to this run's save directory.
    """
    os.makedirs(parent_dir, exist_ok=True)
    experiment_id = 0
    for folder_name in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, folder_name)):
            continue
        try:
            folder_name = int(folder_name.split('-run')[-1])
            if folder_name > experiment_id:
                experiment_id = folder_name
        except:
            pass
    experiment_id += 1

    parent_dir = os.path.join(parent_dir, env_name)
    parent_dir = parent_dir + '-run{}'.format(experiment_id)
    os.makedirs(parent_dir, exist_ok=True)
    return parent_dir



# Search method
class EpsilonTracker:
    def __init__(self, epsilon_start = 1.0, epsilon_final = 0.3, warmup_steps = 0, total_steps = 0, epsilon_decay = 0.995, total_episodes = 100):
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_frames = total_steps - warmup_steps
        self.epsilon_decay = epsilon_decay
        self.epsilon_current = epsilon_start
        self.total_episodes = total_episodes

    def update(self, step_number = 0):

        epsilon = 1.0

        # If no step number mentioned, it means decrease epsilon exponentially based on epsilon decay
        if step_number == 0:
            #self.epsilon_current *= self.epsilon_decay 
            #epsilon = max(self.epsilon_final, self.epsilon_current)
            self.epsilon_current -= (self.epsilon_start - self.epsilon_final)/self.total_episodes
            epsilon = max(self.epsilon_final, self.epsilon_current)
        
        else:
            epsilon = self.epsilon_final + (self.epsilon_start + self.epsilon_final) * math.exp(-1. * step_number/self.epsilon_decay)
            #epsilon = max(self.epsilon_final, self.epsilon_start - step_number / self.epsilon_frames)

        self.epsilon_current = epsilon
            
        return epsilon




class DataVisualization():
    def __init__(self, x_min = 0, x_max = 0, y_min = 0, y_max = 0):
        
        self.plt = plt
        # self.plt.axis([x_min, x_max, y_min, y_max])
        #self.plt.ion()
        #self.plt.show()

    def display(self, tensor):

        #x_grid_label = 

        # data = np.zeros((30, 30))

        # for x in range(tensor.shape[0]):
        #     for y in range(tensor.shape[1]):
        #         if tensor[x][y][0] == 0.01:
        #             data[x][y] = 1

        #         elif tensor[x][y][0] == 1.0:
        #             data[x][y] = 2

        # # create discrete colormap
        # cmap = mcolors.ListedColormap(['white', 'red', 'blue', 'lightyellow', 'yellow'])
        # bounds = [0,1, 2, 3, 4, 5]
        # norm = mcolors.BoundaryNorm(bounds, cmap.N)


        # self.plt.close('all')
        # fig, ax = plt.subplots()

        # im = ax.imshow(data, cmap=cmap, norm=norm)

        # ax.grid(axis='both', linestyle='-', color='k', linewidth=1)
        # ax.set_xticks(np.arange(30))
        # ax.set_yticks(np.arange(30))
        # ax.invert_yaxis()

        # fig.tight_layout()
        # self.plt.tight_layout()
        # self.plt.show()
        # input('Enter to continue: ')
        # self.plt.close()

        vegetables = ["cucumber", "tomato", "lettuce", "asparagus",
              "potato", "wheat", "barley"]
        farmers = ["Farmer Joe", "Upland Bros.", "Smith Gardening",
                   "Agrifun", "Organiculture", "BioGoods Ltd.", "Cornylee Corp."]

        # harvest = np.array([[0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
        #                     [2.4, 0.0, 4.0, 1.0, 2.7, 0.0, 0.0],
        #                     [1.1, 2.4, 0.8, 4.3, 1.9, 4.4, 0.0],
        #                     [0.6, 0.0, 0.3, 0.0, 3.1, 0.0, 0.0],
        #                     [0.7, 1.7, 0.6, 2.6, 2.2, 6.2, 0.0],
        #                     [1.3, 1.2, 0.0, 0.0, 0.0, 3.2, 5.1],
        #                     [0.1, 2.0, 0.0, 1.4, 0.0, 1.9, 6.3]])


        x = np.arange(1,31)
        y = np.arange(1,31)
        x_label = np.char.mod('%d', x)
        y_label = np.char.mod('%d', y)

        data = np.tile(np.array([[1,0],[0,1]]), (15, 15))  #np.resize([1, -1], (30, 30))#np.zeros((30, 30))
        #data = [0, 1] * data

        for x in range(tensor.shape[0]):
            for y in range(tensor.shape[1]):
                if tensor[x][y][0] == 0.01:
                    data[x][y] = 2

                elif tensor[x][y][0] == 1.0:
                    data[x][y] = 3


        # create discrete colormap
        cmap = mcolors.ListedColormap(['white', 'lightgrey', 'red', 'blue', 'lightyellow', 'yellow'])
        bounds = [0,1, 2, 3, 4, 5, 6]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        fig, ax = plt.subplots()
        im = ax.imshow(data, cmap = cmap, norm = norm)

        # We want to show all ticks...
        #ax.grid(axis='both', linestyle='-', color='k', linewidth=1)
        ax.set_xticklabels(x_label)
        ax.set_yticklabels(y_label)
        ax.set_xticks(np.arange(30))
        ax.set_yticks(np.arange(30))
        ax.invert_yaxis()
        # ... and label them with the respective list entries


        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=0, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        # for i in range(30):
        #     for j in range(30):
        #         text = ax.text(i, j, (i),
        #                        ha="center", va="center", color="w")

        ax.set_title("Harvest of local farmers (in tons/year)")
        fig.tight_layout()
        plt.show()

        input('Enter to continue: ')
        self.plt.close()

    def show(self):
        self.show()