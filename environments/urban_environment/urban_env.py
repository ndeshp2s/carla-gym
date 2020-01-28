
import sys, glob, os, random, time
import math
import numpy as np
from environments.carla_gym import CarlaGym
from environments.urban_environment import carla_config
from environments.urban_environment.actions import ACTIONS, ACTION_CONTROL, ACTIONS_NAMES
#from environments.urban_environment.spawner import Spawner
#from environments.urban_environment.walker_spawner import Walkers

import gym
from gym import spaces

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
from carla import ColorConverter as cc
from agents.tools.misc import get_speed

from utils.renderer import Renderer
from utils.planner import Planner
import agents.navigation.cutils


DEBUG = 1

try:
    import pygame
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')


class UrbanEnv(CarlaGym):
    """docstring for ClassName"""
    def __init__(self):
        super(UrbanEnv, self).__init__()

        # Initialize environment parameters
        self.town = "Town01"

        # Sensors
        self.rgb_sensor = carla_config.rgb_sensor
        self.sem_sensor = carla_config.sem_sensor

        
        # Rendering related
        # self.is_render_enabled = carla_config.render

        # if self.is_render_enabled:
        #     self.renderer = Renderer()     
        #     self.init_renderer()


        # Simulating pedestrians
        self.walkers = None

        self.target_speed_prev = 0.0

        low = np.array([])
        high = np.array([])
        for i in range(carla_config.num_of_ped):
            np.concatenate((low, np.array([np.finfo(np.float32).min, np.finfo(np.float32).min, -360, 0])))
            np.concatenate((high, np.array([np.finfo(np.float32).min, np.finfo(np.float32).min, 360, 0])))


        self.action_space = spaces.Discrete(carla_config.N_DISCRETE_ACTIONS)

        self.observation_space = spaces.Box(low=0, high=255, shape = (carla_config.WIDTH, carla_config.HEIGHT, carla_config.N_CHANNELS), dtype=np.uint8)

    def step(self, action = None,sp=25):

        self._take_action(action, sp)

        self.tick()

        if self.render: 
            if self.rgb_image is not None:
                img = self.get_rendered_image()
                self.renderer.render_image(img)

        obs = self._get_state()

        print('state: ', obs)

        w = self.world.get_map().get_waypoint(self.ego_vehicle.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving | carla.LaneType.Shoulder | carla.LaneType.Sidewalk)

        reward, done = self._get_reward()

        return np.asarray(obs), reward, done, {}


    def _get_state(self):

        state = [0 for i in range(self.observation_space.shape[0])]
        actor_list = self.world.get_actors()
        pedestrian_list = actor_list.filter('walker.pedestrian.*')


        ped_list_sorted = []

        for p in pedestrian_list:
            pos = self.ego_vehicle.get_transform().transform(p.get_transform().location)
            ped_info = (p.id, math.sqrt(pos.x*pos.x + pos.y*pos.y))
            ped_list_sorted.append(ped_info)


        ped_list_sorted.sort(key = lambda x:x[1])

        counter = 0
        for p in ped_list_sorted:
            ped = self.world.get_actor(p[0])
            

            ev_loc = self.ego_vehicle.get_transform()
            ped_loc = ped.get_transform().location

            rel_ped_pos = self.ego_vehicle.get_transform().transform(ped_loc)
            rel_ped_head = ped.get_transform().rotation.yaw - self.ego_vehicle.get_transform().rotation.yaw

            if math.sqrt(rel_ped_pos.x*rel_ped_pos.x + rel_ped_pos.y*rel_ped_pos.y) > 5000:
                continue

            w = self.world.get_map().get_waypoint(ped_loc, project_to_road=True, lane_type=carla.LaneType.Driving | carla.LaneType.Sidewalk)
            l_type = 1
            if w.lane_type == 'Driving':
                l_type = 0
            elif w.lane_type == 'Sidewalk':
                l_type = 1

            state[counter] = rel_ped_pos.x
            counter += 1
            state[counter] = rel_ped_pos.y
            counter += 1
            state[counter] = rel_ped_head
            counter += 1
            state[counter] = l_type
            counter += 1

            print('counter, state', counter, len(state))
            if counter >= len(state):
                break
        
        return state


    def _get_reward(self):

        total_reward = d_reward = nc_reward = c_reward = 0

        # Reward for speed 

        ev_speed = self.get_ego_speed()

        if ev_speed > 0.0 and ev_speed <=50:
            d_reward = (10.0 - abs(10.0 - ev_speed))/10.0

        elif ev_speed > 50:
            d_reward = -5

        elif ev_speed <= 0.0:
            d_reward = -2


        # Reward for safety
        ped_list = self.world.get_actors().filter('walker.pedestrian.*')

        total_reward = d_reward + nc_reward + c_reward
        return 0, False

    def get_ego_speed(self):
        return get_speed(self.ego_vehicle)

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


    def cruise(self):
        self.planner.local_planner.set_speed(25)
        self.apply_control(control)

    def apply_control(self, target_speed):

        if target_speed is not None:
            self.planner.local_planner.set_speed(target_speed)
        control = self.planner.run_step()
        self.ego_vehicle.apply_control(control)


    def emergency_stop(self):
        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 1.0
        control.hand_brake = False
        self.ego_vehicle.apply_control(control)

    def slow_down(self):
        current_speed = get_speed(self.ego_vehicle)
        self.planner.local_planner.set_speed(current_speed - 1.0)
        control = self.planner.run_step()
        self.ego_vehicle.apply_control(control)
        control.brake = 0.0

    def stop(self):
        
        self.planner.local_planner.set_speed(0)
        control = self.planner.run_step()
        control.brake = -0.05
        self.ego_vehicle.apply_control(control)
        print('control stop: ', control.brake)



    def reset(self, client_only = False):

        self.setup_client_and_server(client_only)

        # self.initialize_ego_vehicle()

        # self.apply_settings()

        # # Get rid of initial moment issue
        # for i in range(25):
        #     self.step('2')

        # obs = self._get_state()

        # return np.asarray(obs)


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