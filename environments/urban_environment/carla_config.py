town = "Town11"
sp_x = 40.0
sp_y = -1.8
sp_z = 0.2
sp_yaw = 180.0
sp_pitch = 0.0
sp_roll = 0.0

ev_bp = 'vehicle.audi.etron'
ev_name = 'hero'
ev_goal_x = -132.0
ev_goal_y = 48.25
ev_goal_z = 0.4
ev_goal_yaw = -180.0

# Sensors
rgb_sensor = True
rgb_size_x = '1920'
rgb_size_y = '1080'
rgb_fov = '110'
rgb_loc_x = 2.5
rgb_loc_y = 0.0
rgb_loc_z = 2.0

sem_sensor = False

# Display related
display = False

# Rendering related
render = False
screen_x = 720
screen_y = 720


#ACTIONS = ['forward', 'brake', 'no_action'] #['forward', 'forward_left', 'forward_right', 'brake', 'brake_left', 'brake_right']
ACTIONS = ['accelerate', 'cont', 'decelerate', 'brake']
N_DISCRETE_ACTIONS = 4

# Observation space
num_of_ped = 3
HEIGHT = 1
WIDTH = 4*num_of_ped
N_CHANNELS = 0
grid_height = 60
grid_width = 30
features = 3
lane_types = 3# Driving(road)-1, sidewalk(shoulder)-2, crosswalk-3 

x_min = -5
x_max = 35
x_size = 30
y_min = -15
y_max = 15
y_size = 30

# Pedestrian behavior probalities
# Four types of behaviors:
## normal walking
## normal crossing at intersection
## Jaw-walking
## standing on road (group of people standing on road side)
## pedestrian walking on road
ped_beh_prob = [1.0, 0.0, 0.0, 0.0,0.0] # 

# Spawner related
num_of_ped = 6
num_of_veh = 0
percentage_pedestrians_crossing = 0.8
percentage_pedestrians_crossing_illegal = 0.2
ped_spawn_min_dist = 15.0
ped_spawn_max_dist = 40.0
ped_max_dist = 50.0
ped_min_dist = 0.0