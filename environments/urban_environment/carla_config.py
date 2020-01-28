town = "Town-01"
sp_x = 220.0
sp_y = 129.5
sp_z = 0.1
sp_yaw = 180.0
sp_pitch = 0.0
sp_roll = 0.0

eg_bp = 'vehicle.audi.etron'
eg_name = 'hero'

# Sensors
rgb_sensor = True
rgb_size_x = '1920'
rgb_size_y = '1080'
rgb_fov = '110'
rgb_loc_x = 2.5
rgb_loc_y = 0.0
rgb_loc_z = 2.0

sem_sensor = False



# Rendering related
render = True
screen_x = 720
screen_y = 720


#ACTIONS = ['forward', 'brake', 'no_action'] #['forward', 'forward_left', 'forward_right', 'brake', 'brake_left', 'brake_right']
ACTIONS = ['accelerate', 'cont', 'decelerate', 'brake']
N_DISCRETE_ACTIONS = 3

# Observation space
num_of_ped = 3
HEIGHT = 1
WIDTH = 4*num_of_ped
N_CHANNELS = 0

# Pedestrian behavior probalities
# Four types of behaviors:
## normal walking
## normal crossing at intersection
## Jaw-walking
## standing on road (group of people standing on road side)
## pedestrian walking on road
ped_beh_prob = [1.0, 0.0, 0.0, 0.0,0.0] # 