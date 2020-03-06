import sys
import glob
import os
import random
import numpy as np

import gym
from gym import spaces

try:
    sys.path.append(glob.glob('/home/niranjan/carla/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla


#from agents.tools.misc import get_speed
import agents
from environments.carla_gym import CarlaGym
from environments.urban_environment import carla_config
from utils.renderer import Renderer
from utils.planner import Planner

class UrbanEnv(CarlaGym):
    """docstring for ClassName"""
    def __init__(self):
        super(UrbanEnv, self).__init__()

        # Initialize environment parameters
        self.town = carla_config.town
        self.state_y = carla_config.grid_height
        self.state_x = carla_config.grid_width
        self.channel = carla_config.features

        # Sensors
        self.rgb_sensor = carla_config.rgb_sensor
        self.sem_sensor = carla_config.sem_sensor

        # Planners
        self.planner = None

        # States and actions
        self.observation_space = spaces.Box(low=0, high=255, shape=(carla_config.grid_height, carla_config.grid_width, carla_config.features), dtype=np.uint8)
        self.action_space = spaces.Discrete(carla_config.N_DISCRETE_ACTIONS)

        
        # Rendering related
        self.renderer = None
        self.is_render_enabled = carla_config.render

        if self.is_render_enabled:
            self.renderer = Renderer()     
            self.init_renderer()


    def step(self, action = None,sp=25):

        self._take_action(action, sp)

        self.tick()

        state = self._get_observation()

        reward, done = self._get_reward()

        if self.render: 
            if self.rgb_image is not None:
                img = self.get_rendered_image()
                self.renderer.render_image(img)

        return state, reward, done 


    def _get_observation(self):
        o = np.zeros([self.state_y, self.state_x, self.channel])
        return o


    def _get_reward(self):

        done = False

        total_reward = d_reward = nc_reward = c_reward = 0

        # # Reward for speed 
        # ev_speed = get_speed(self.ego_vehicle)

        # if ev_speed > 0.0 and ev_speed <=50:
        #     d_reward = (10.0 - abs(10.0 - ev_speed))/10.0

        # elif ev_speed > 50:
        #     d_reward = -5

        # elif ev_speed <= 0.0:
        #     d_reward = -2


        total_reward = d_reward + nc_reward + c_reward

        return total_reward, done


    def _take_action(self, action, sp):

        mps_to_kmph = 3.6

        target_speed = 0.0

        if action == '0': # speed tracking 

            self.planner.local_planner.set_speed(25.0)
            control = self.planner.run_step()
            control.brake = 0.0
            self.ego_vehicle.apply_control(control)

        elif action == '1': # slow down/decelerate
            current_speed = get_speed(self.ego_vehicle)
            self.planner.local_planner.set_speed(current_speed - 1)
            control = self.planner.run_step()
            control.brake = 0.0
            self.ego_vehicle.apply_control(control)


        elif action == '2': # stop
            self.planner.local_planner.set_speed(0)
            control = self.planner.run_step()
            self.ego_vehicle.apply_control(control)

        elif action == '3': # emergency stop
            self.emergency_stop()


    def emergency_stop(self):
        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 1.0
        control.hand_brake = False
        self.ego_vehicle.apply_control(control)


    def reset(self, client_only = False):

        self.setup_client_and_server(client_only)

        self.initialize_ego_vehicle()

        self.apply_settings()


    def initialize_ego_vehicle(self):
        # Spawn ego vehicle
        sp = carla.Transform(carla.Location(x=carla_config.sp_x, y=carla_config.sp_y, z=carla_config.sp_z), 
                             carla.Rotation(yaw=carla_config.sp_yaw))
        bp = random.choice(self.world.get_blueprint_library().filter(carla_config.eg_bp))
        bp.set_attribute('role_name', carla_config.eg_name)
        self.ego_vehicle = self.spawn_ego_vehicle(bp, sp)

        # Add sensors to ego vehicle
        # RGB sensor
        if carla_config.rgb_sensor:
            rgb_bp = rgb_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
            rgb_bp.set_attribute('image_size_x', carla_config.rgb_size_x)
            rgb_bp.set_attribute('image_size_y', carla_config.rgb_size_y)
            rgb_bp.set_attribute('fov', carla_config.rgb_fov)
            transform = carla.Transform(carla.Location(x=carla_config.rgb_loc_x, z=carla_config.rgb_loc_z))
            self.rgb_sensor = self.world.spawn_actor(rgb_bp, transform, attach_to = self.ego_vehicle)
            self.rgb_sensor.listen(self.rgb_sensor_callback)
            self.rgb_image = None

        # Semantic sensor
        if carla_config.sem_sensor:
            sem_bp = self.world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
            sem_bp.set_attribute('image_size_x', carla_config.rgb_size_x)
            sem_bp.set_attribute('image_size_y', carla_config.rgb_size_y)
            sem_bp.set_attribute('fov', carla_config.rgb_fov)
            trnasform = carla.Transform(carla.Location(x=carla_config.rgb_loc_x, z=carla_config.rgb_loc_z))
            self.semantic_sensor = self.world.spawn_actor(sem_bp, trnasform, attach_to=self.ego_vehicle)
            self.semantic_sensor.listen(self.semantic_sensor_callback)
            self.semantic_image = None

        # Initialize the planner
        self.planner = Planner()
        self.planner.initialize(self.ego_vehicle)
        spawn_point = self.world.get_map().get_spawn_points()[0] #
        self.planner.set_destination((spawn_point.location.x, spawn_point.location.y, spawn_point.location.z))

    def spawn_ego_vehicle(self, bp = None, sp = None):

        if not bp:
            bp = random.choice(self.world.get_blueprint_library().filter('vehicle.*'))

        if not sp:
            sp =  random.choice(self.world.get_map().get_spawn_points())

        return self.world.spawn_actor(bp, sp)


    def rgb_sensor_callback(self, image):
        #image.convert(cc.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        self.rgb_image = array

    def semantic_sensor_callback(self, image):
        image.convert(cc.CityScapesPalette)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        self.semantic_image = array


    def close(self):
        if carla_config.rgb_sensor:
            self.rgb_sensor.destroy()
        if carla_config.sem_sensor:
            self.semantic_sensor.destroy()
        if self.ego_vehicle is not None:
            self.ego_vehicle.destroy()
            
        if self.renderer is not None:
            self.renderer.close()
        self.kill_processes()
        


    def init_renderer(self):
        self.no_of_cam = 0
        if self.rgb_sensor: self.no_of_cam += 1
        if self.sem_sensor: self.no_of_cam += 1

        self.renderer.create_screen(carla_config.screen_x, carla_config.screen_y*self.no_of_cam)

    def render(self):
        if self.renderer is None or not (self.is_render_enabled):
            return

        img =  self.get_rendered_image()
        self.renderer.render_image(img)

    def get_rendered_image(self):
        temp = []
        if self.rgb_sensor: temp.append(self.rgb_image)
        if self.sem_sensor: temp.append(self.semantic_image)

        return np.vstack(img for img in temp)
