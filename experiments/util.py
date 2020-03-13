
import os
import math


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
    def __init__(self, epsilon_start = 1.0, epsilon_final = 0.1, warmup_steps = 0, total_steps = 0, epsilon_decay = 0.995):
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_frames = total_steps - warmup_steps
        self.epsilon_decay = epsilon_decay
        self.epsilon_current = epsilon_start

    def update(self, step_number = 0):

        epsilon = 1.0

        # If no step number mentioned, it means decrease epsilon exponentially based on epsilon decay
        if step_number == 0:
            self.epsilon_current *= self.epsilon_decay 
            epsilon = max(self.epsilon_final, self.epsilon_current)
        
        else:
            epsilon = self.epsilon_final + (self.epsilon_start + self.epsilon_final) * math.exp(-1. * step_number/self.epsilon_decay)
            #epsilon = max(self.epsilon_final, self.epsilon_start - step_number / self.epsilon_frames)

        self.epsilon_current = epsilon
            
        return epsilon