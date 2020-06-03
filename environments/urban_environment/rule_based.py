import sys
import glob
import os
import random
import numpy as np
import gym

try:
    sys.path.append(glob.glob('/home/niranjan/carla/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla

from agents.navigation.basic_agent import BasicAgent
from agents.tools.misc import get_speed

from environments.carla_gym import CarlaGym
from environments.urban_environment import carla_config
from utils.renderer import Renderer



class RuleBasedEnv(CarlaGym):
    """docstring for ClassName"""
    def __init__(self):
        super(RuleBasedEnv, self).__init__()

        # Initialize environment parameters
        self.town = carla_config.town
        self.state_y = carla_config.grid_height
        self.state_x = carla_config.grid_width
        self.channel = carla_config.features

        # Sensors
        self.rgb_sensor = None
        self.semantic_sensor = None

        # Basic Agent
        self.basic_agent = None
        self.ego_vehicle = None
            
        # Rendering related
        self.renderer = None
        self.is_render_enabled = carla_config.render

        if self.is_render_enabled:
            self.renderer = Renderer()     
            self.init_renderer()


    def step(self):

        control = self.basic_agent.run_step()
        self.ego_vehicle.apply_control(control)
        self.tick()

        if self.render and self.is_render_enabled: 
            if self.rgb_image is not None:
                img = self.get_rendered_image()
                self.renderer.render_image(img)


    def reset(self, client_only = False):

        if self.server:
            self.close() 

        self.setup_client_and_server(display = carla_config.display, rendering = carla_config.render, reconnect_client_only = client_only)

        self.initialize_ego_vehicle()

        self.apply_settings(fps = 10.0, no_rendering = not carla_config.render)

        self.world.get_map().generate_waypoints(1.0)



    def initialize_ego_vehicle(self):
        # Spawn ego vehicle
        sp = carla.Transform(carla.Location(x = carla_config.sp_x, y = carla_config.sp_y, z = carla_config.sp_z), 
                             carla.Rotation(yaw = carla_config.sp_yaw))
        bp = random.choice(self.world.get_blueprint_library().filter(carla_config.ev_bp))
        bp.set_attribute('role_name', carla_config.ev_name)
        self.ego_vehicle = self.spawn_ego_vehicle(bp, sp)

        # Add sensors to ego vehicle
        # RGB sensor
        if carla_config.rgb_sensor:
            rgb_bp = rgb_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
            rgb_bp.set_attribute('image_size_x', carla_config.rgb_size_x)
            rgb_bp.set_attribute('image_size_y', carla_config.rgb_size_y)
            rgb_bp.set_attribute('fov', carla_config.rgb_fov)
            transform = carla.Transform(carla.Location(x = carla_config.rgb_loc_x, z = carla_config.rgb_loc_z))
            self.rgb_sensor = self.world.spawn_actor(rgb_bp, transform, attach_to = self.ego_vehicle)
            self.rgb_sensor.listen(self.rgb_sensor_callback)
            self.rgb_image = None

        # Semantic sensor
        if carla_config.sem_sensor:
            sem_bp = self.world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
            sem_bp.set_attribute('image_size_x', carla_config.rgb_size_x)
            sem_bp.set_attribute('image_size_y', carla_config.rgb_size_y)
            sem_bp.set_attribute('fov', carla_config.rgb_fov)
            trnasform = carla.Transform(carla.Location(x = carla_config.rgb_loc_x, z = carla_config.rgb_loc_z))
            self.semantic_sensor = self.world.spawn_actor(sem_bp, trnasform, attach_to = self.ego_vehicle)
            self.semantic_sensor.listen(self.semantic_sensor_callback)
            self.semantic_image = None

        # Initialize the Basic Agent
        # self.planner = Planner()
        # self.planner.initialize(self.ego_vehicle)
        # self.planner.set_destination((carla_config.ev_goal_x, carla_config.ev_goal_y, carla_config.ev_goal_z))
        self.basic_agent = BasicAgent(vehicle = self.ego_vehicle, target_speed = 15)
        self.basic_agent.set_destination((carla_config.ev_goal_x, carla_config.ev_goal_y, carla_config.ev_goal_z))


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
        array = np.frombuffer(image.raw_data, dtype = np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        self.semantic_image = array


    def close(self):
        #if carla_config.rgb_sensor:
        if self.rgb_sensor is not None:
            self.rgb_sensor.destroy()
        #if carla_config.sem_sensor:
        if self.semantic_sensor is not None:
            self.semantic_sensor.destroy()
        # if self.ego_vehicle is not None:
        #     self.ego_vehicle.destroy()
            
        if self.renderer is not None:
            self.renderer.close()
        self.kill_processes()
        

    def init_renderer(self):
        self.no_of_cam = 0
        if carla_config.rgb_sensor: self.no_of_cam += 1
        if carla_config.sem_sensor: self.no_of_cam += 1

        self.renderer.create_screen(carla_config.screen_x, carla_config.screen_y * self.no_of_cam)


    def render(self):
        if self.renderer is None or not (self.is_render_enabled):
            return

        img =  self.get_rendered_image()
        self.renderer.render_image(img)


    def get_rendered_image(self):
        temp = []
        if carla_config.rgb_sensor: temp.append(self.rgb_image)
        if carla_config.sem_sensor: temp.append(self.semantic_image)

        return np.vstack(img for img in temp)


    def get_ego_speed(self):
        ev_speed = get_speed(self.ego_vehicle)
        ev_speed = round(ev_speed, 2)

        if ev_speed <= 0.0:
            ev_speed = 0.0
        
        return ev_speed

