town = "Town11"
sp_x = 34.0
sp_y = -1.8
sp_z = 0.2
sp_yaw = 180.0
sp_pitch = 0.0
sp_roll = 0.0

ev_bp = 'vehicle.audi.etron'
ev_name = 'hero'
ev_goal_x = -140.0
ev_goal_y = 40.25
ev_goal_z = 0.2
ev_goal_yaw = 180.0

# Sensors
rgb_sensor = True
rgb_size_x = '1920'
rgb_size_y = '1080'
rgb_fov = '110'
rgb_loc_x = -6.5#2.5
rgb_loc_y = 0.0
rgb_loc_z = 4.0#2.0

sem_sensor = False

# Display related
display = True

# Rendering related
render = True
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
grid_height = 32
grid_width = 32
features = 4
lane_types = 3# Driving(road)-1, sidewalk(shoulder)-2, crosswalk-3 

x_min = 0
x_max = 32
x_size = 32
y_min = -16
y_max = 16
y_size = 32

# Pedestrian behavior probalities
# Four types of behaviors:
## normal walking
## normal crossing at intersection
## Jaw-walking
## standing on road (group of people standing on road side)
## pedestrian walking on road
ped_beh_prob = [1.0, 0.0, 0.0, 0.0,0.0] # 
ped_max_speed = 5.0

# Spawner related
num_of_ped = 6
num_of_veh = 0
percentage_pedestrians_crossing = 0.6
percentage_pedestrians_crossing_illegal = 0.2
ped_spawn_min_dist = 5.0
ped_spawn_max_dist = 30.0
ped_max_dist = 31.0
ped_min_dist = 0.0