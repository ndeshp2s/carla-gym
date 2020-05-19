import sys
import glob
import os
import random
import numpy as np
import math
import transforms3d

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


from agents.tools.misc import get_speed, is_within_distance_ahead, distance_vehicle
from environments.carla_gym import CarlaGym
from environments.urban_environment import carla_config
from environments.urban_environment.crosswalk_zones import within_crosswalk
from utils.renderer import Renderer
from utils.planner import Planner
from utils.miscellaneous import pedestrian_relative_position


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
        self.rgb_sensor = None
        self.semantic_sensor = None

        # Planners
        self.planner = None

        # States and actions
        self.observation_space = spaces.Box(low = 0, high = 255, shape = (carla_config.grid_height, carla_config.grid_width, carla_config.features), dtype = np.uint8)
        self.action_space = spaces.Discrete(carla_config.N_DISCRETE_ACTIONS)

        self.ego_vehicle = None
        self.current_speed = 0.0
        self.previous_speed = 0.0
        self.max_allowed_speed = 20.0
        self.max_reachable_speed = 25.0

        
        # Rendering related
        self.renderer = None
        self.is_render_enabled = carla_config.render

        if self.is_render_enabled:
            self.renderer = Renderer()     
            self.init_renderer()


    def step(self, action = None,sp=25, model_output = None):

        if action is not None:
            self._take_action(action, sp)

        self.tick()

        state = self._get_observation(action)

        reward, done, info = self._get_reward()

        if self.render and self.is_render_enabled: 
            if self.rgb_image is not None:
                img = self.get_rendered_image()
                self.renderer.render_image(img, model_output)

        return state, reward, done, info 


    def _get_observation(self, action = 0):
        tensor1 = np.zeros([self.state_y, self.state_x, self.channel])
        tensor2 = np.zeros([2])

        # Fill ego vehicle information
        ev_trans = self.ego_vehicle.get_transform()

        ev_speed = get_speed(self.ego_vehicle)
        if ev_speed < 0.0:
            ev_speed = 0.0
        ev_speed = round(ev_speed, 2)
        tensor2[0] = self.normalize_data(ev_speed, 0.0, self.max_reachable_speed)
        
        ev_head = (ev_trans.rotation.yaw + 360) % 360
        ev_head = round(ev_head, 2)
        ev_head_norm = self.normalize_data(ev_head, 0.0, 360.0)
        ev_head_norm =round(ev_head_norm, 2)
        tensor2[1] = ev_head_norm
        

        # Fill pedestrian information
        peds = self.world.get_actors().filter('walker.pedestrian.*')
        for p in peds:
            p_trans = p.get_transform()
            #print(p_trans.rotation.yaw - ev_trans.rotation.yaw)
            
            p_xyz = np.array([p_trans.location.x, p_trans.location.y, p_trans.location.z])
            ev_xyz =  np.array([ev_trans.location.x, ev_trans.location.y, ev_trans.location.z])
            ped_loc = p_xyz - ev_xyz

            pitch = math.radians(ev_trans.rotation.pitch)
            roll = math.radians(ev_trans.rotation.roll)
            yaw = math.radians(ev_trans.rotation.yaw)
            R = transforms3d.euler.euler2mat(roll, pitch, yaw).T
            ped_loc_relative = np.dot(R, ped_loc)

            x_discrete, status_x = self.get_index(ped_loc_relative[0], carla_config.x_min, carla_config.x_max, carla_config.x_size)
            y_discrete, status_y = self.get_index(ped_loc_relative[1], carla_config.y_min, carla_config.y_max, carla_config.y_size)


            if status_x and status_y:
                x_discrete = np.argmax(x_discrete)
                y_discrete = np.argmax(y_discrete)

                p_heading = (p_trans.rotation.yaw + 360) % 360
                p_heading = round(p_heading, 2)
                ev_heading = (ev_trans.rotation.yaw + 360) % 360
                ev_heading = round(ev_heading, 2)

                p_relative_heading = p_heading - ev_heading
                p_relative_heading = (p_relative_heading + 360) % 360
                p_relative_heading = round(p_relative_heading, 2)

                p_speed = get_speed(p)
                p_speed = round(p_speed, 2)


                # # Get pdestrian id
                # p_id = int(p.attributes['role_name'])

                # # Get pedestrian relative orientation
                # p_heading = p_trans.rotation.yaw - ev_trans.rotation.yaw

                # Get pedestrian lane type
                ped_lane = None
                waypoint = self.world.get_map().get_waypoint(p_trans.location, project_to_road=True, lane_type=(carla.LaneType.Driving | carla.LaneType.Sidewalk))
                if waypoint.lane_type == carla.LaneType.Driving:
                    ped_lane = 1
                elif waypoint.lane_type == carla.LaneType.Sidewalk or waypoint.lane_type == carla.LaneType.Shoulder:
                    ped_lane = 2

                if within_crosswalk(p_trans.location.x, p_trans.location.y):
                    ped_lane = 3


                # tensor1[x_discrete, y_discrete,:] = [self.normalize_data(p_id, 0.0, carla_config.num_of_ped), 
                #                                     self.normalize_data(p_heading, 0.0, 360.0),       
                #                                     self.normalize_data(ped_lane, 0.0, 3)]
                #                                     # ped id, ped relative orientation and region occupied.

                #print(tensor[x_discrete, y_discrete, :])
                p_relative_heading_norm = self.normalize_data(p_relative_heading, 0.0, 360.0)
                p_relative_heading_norm = round(p_relative_heading_norm, 2)
                p_speed_norm = self.normalize_data(p_speed, 0.0, carla_config.ped_max_speed)
                p_speed_norm = round(p_speed_norm, 2)

                p_lane = self.normalize_data(ped_lane, 0.0, 3)

                tensor1[x_discrete, y_discrete,:] = [1.0, p_relative_heading_norm, p_speed_norm, p_lane]

        tensor = []
        tensor.append(tensor1)
        tensor.append(tensor2)
        
        return tensor


    def _get_reward(self):

        done = False
        info = 'Done'

        total_reward = d_reward = nc_reward = c_reward = 0.0

        # Reward for speed 
        ev_speed = get_speed(self.ego_vehicle)
        if ev_speed < 0.0:
            ev_speed = 0.0
        ev_speed = round(ev_speed, 2)

        if ev_speed > 0.0 and ev_speed <= self.max_allowed_speed:
            d_reward = (self.max_allowed_speed - abs(self.max_allowed_speed - ev_speed))/self.max_allowed_speed

        elif ev_speed > self.max_allowed_speed:
            d_reward = -1.0

        elif ev_speed <= 0.0:
            d_reward = -1.0
        
        ## Reward(penalty) for collision
        pedestrian_list = self.world.get_actors().filter('walker.pedestrian.*')
        collision, near_collision, pedestrian = self.is_collision(pedestrian_list)

        if collision is True and ev_speed > 0.0:
            #print('collision')
            c_reward = -10
            done = True
            info = 'Collision'

        elif collision is True:
            done = True
            info = 'Ped-Collision'

        elif near_collision is True and ev_speed > 0.0:
            #print('near collision')
            nc_reward = -2

        # Check if goal reached
        if self.planner.done():
            done = True
            info = 'Goal Reached'
        # ev_trans = self.ego_vehicle.get_transform()

        # d = self.distance(ev_trans, carla.Transform(carla.Location(x = carla_config.ev_goal_x, y = carla_config.ev_goal_y, z = carla_config.ev_goal_z)))
        # if d < 6:
        #     done = True


        total_reward = d_reward + nc_reward + c_reward
        total_reward = round(total_reward, 2)

        return total_reward, done, info


    def _take_action(self, action, sp):
                # control = self.planner.run_step()
        # self.ego_vehicle.apply_control(control)

        # self.current_speed = get_speed(self.ego_vehicle)

        # if action == 0:
        #     desired_speed = 0
        #     if self.current_speed < self.previous_speed:
        #         desired_speed = self.previous_speed + 1.0
        #     elif self.current_speed > self.previous_speed:
        #         desired_speed = self.current_speed + 1.0

        # elif action == 1:
        #     desired_speed = 0
        #     if self.current_speed > self.previous_speed:
        #         desired_speed = self.previous_speed - 1.0
        #     elif self.current_speed < self.previous_speed:
        #         desired_speed = self.current_speed - 1.0

        # elif action == 2:
        #     desired_speed = 0

        # elif action == 3:
        #     desired_speed = -1


        # self.planner.local_planner.set_speed(desired_speed)
        # control = self.planner.run_step()
        # control.brake = 0.0
        # self.ego_vehicle.apply_control(control)

        # print(self.current_speed, self.previous_speed, desired_speed)
        # if desired_speed != -1:
        #     self.previous_speed = desired_speed





        # desired_speed = self.planner.local_planner.get_target_speed()
        # if action == 1:
        #     desired_speed += 1
        #     self.planner.local_planner.set_speed(1)
        #     control = self.planner.run_step()
        #     control.brake = 0.0
        #     control.throttle = round(control.throttle, 2)
        #     self.ego_vehicle.apply_control(control)

        # elif action == 2:
        #     desired_speed -= 1
        #     self.planner.local_planner.set_speed(-1)
        #     control = self.planner.run_step()
        #     control.brake = 0.0
        #     control.throttle = round(control.throttle, 2)
        #     self.ego_vehicle.apply_control(control)

        # elif action == 3:
        #     self.planner.local_planner.set_speed(0)
        #     control = self.planner.run_step()
        #     control.brake = 1.0
        #     control.throttle = round(control.throttle, 2)
        #     self.ego_vehicle.apply_control(control)

        # elif action == 0:
        #     control = self.planner.run_step()
        #     control.brake = 0.0
        #     if self.get_ego_speed() < 1.0:
        #         control.brake = 1
        #     control.throttle = round(control.throttle, 2)
        #     self.ego_vehicle.apply_control(control)
        # control = self.planner.run_step()
        # self.ego_vehicle.apply_control(control)

        #mps_kmph_conversion = 3.6

        # target_speed = 0.0

        # accelerate
        if action == 0:
            self.planner.local_planner.set_speed(1)
            control = self.planner.run_step()
            control.brake = 0.0
            control.throttle = round(control.throttle, 2)
            self.ego_vehicle.apply_control(control)

        elif action == 1:
            self.planner.local_planner.set_speed(-1)
            control = self.planner.run_step()
            control.brake = 0.0
            control.throttle = round(control.throttle, 2)
            self.ego_vehicle.apply_control(control)

        elif action == 2:
            self.planner.local_planner.set_speed(0)
            control = self.planner.run_step()
            control.brake = 1.0
            control.throttle = round(control.throttle, 2)
            self.ego_vehicle.apply_control(control)

        elif action == 3:
            control = self.planner.run_step()
            control.brake = 0.0
            if self.get_ego_speed() < 1.0:
                control.brake = 1
            control.throttle = round(control.throttle, 2)
            self.ego_vehicle.apply_control(control)


        # # decelerate
        # elif action == 2:
        #     current_speed = self.get_ego_speed()
        #     desired_speed = current_speed - 1.0
        #     #desired_speed *= 3.6
        #     self.current_speed = desired_speed
        #     self.planner.local_planner.set_speed(round(desired_speed, 2))
        #     control = self.planner.run_step()
        #     control.brake = 0.0
        #     self.ego_vehicle.apply_control(control)


        # elif action == 3: # emergency stop
        #     self.current_speed = 0
        #     self.emergency_stop()

        
        # # speed tracking
        # elif action == 0:
        #     #self.planner.local_planner.set_speed(self.current_speed)
        #     control = self.planner.run_step()
        #     control.brake = 0.0
        #     self.ego_vehicle.apply_control(control)


    def emergency_stop(self):
        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 1.0
        control.hand_brake = False
        self.ego_vehicle.apply_control(control)


    def reset(self, client_only = False):

        if self.server:
            self.close() 

        self.setup_client_and_server(display = carla_config.display, rendering = carla_config.render)

        self.initialize_ego_vehicle()

        self.apply_settings(fps = 10.0, no_rendering = not carla_config.render)

        self.world.get_map().generate_waypoints(1.0)

        # Run some initial steps
        for i in range(10):
            self.step(0)
        for i in range(10):
            self.step(2)       
        # for i in range(10):
        #     self.step(3)

        state = self._get_observation()

        return state


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

        # Initialize the planner
        self.planner = Planner()
        self.planner.initialize(self.ego_vehicle)
        self.planner.set_destination((carla_config.ev_goal_x, carla_config.ev_goal_y, carla_config.ev_goal_z))


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


    def get_index(self, val, start, stop, num):

        grids = np.linspace(start, stop, num)
        features = np.zeros(num)

        #Check extremes
        if val <= grids[0] or val > grids[-1]:
            return features, False

        for i in range(len(grids) - 1):
            if val >= grids[i] and val < grids[i + 1]:
                features[i] = 1

        return features, True


    def normalize_data(self, data, min_val, max_val):
        return (data - min_val)/(max_val - min_val)




    def is_collision(self, entity_list):

        ego_vehicle_location = self.ego_vehicle.get_transform().location
        ego_vehicle_waypoint = self.world.get_map().get_waypoint(ego_vehicle_location)


        for target in entity_list:

            # if the object is not in our lane it's not an obstacle
            target_waypoint = self.world.get_map().get_waypoint(target.get_location(), project_to_road = True, lane_type = (carla.LaneType.Driving | carla.LaneType.Sidewalk))

            ped_loc = pedestrian_relative_position(ped_trans = target.get_transform(), ev_trans = self.ego_vehicle.get_transform())
            ped_loc[0] = round(ped_loc[0], 2)
            ped_loc[1] = round(ped_loc[1], 2)

            # Check for Collision
            if target_waypoint.lane_type == carla.LaneType.Driving and target_waypoint.lane_id == ego_vehicle_waypoint.lane_id: 
                if (ped_loc[0] <= 3.5 and ped_loc[0] >= -2.2) and abs(ped_loc[1]) <= 2.2:
                    return (True, True, target)

            # Check for near collision
            if target_waypoint.lane_type == carla.LaneType.Driving:
                if ped_loc[0] <= 8.0 and ped_loc[0] >= -2.2: 
                    return (False, True, target)

            # # if target_waypoint.road_id == ego_vehicle_waypoint.road_id and \
            # #         target_waypoint.lane_id == ego_vehicle_waypoint.lane_id:
            #         #target_waypoint.lane_type == ego_vehicle_waypoint.lane_type:
            # if target_waypoint.lane_type == carla.LaneType.Driving and target_waypoint.lane_id == ego_vehicle_waypoint.lane_id:
            #     #if is_within_distance_ahead(target.get_transform(), self.ego_vehicle.get_transform(), 4.0):
            #     #if self.distance(self.ego_vehicle.get_transform(), target.get_transform()) < 10.0:
            #     ped_loc = pedestrian_relative_position(ped_trans = target.get_transform(), ev_trans = self.ego_vehicle.get_transform())

            #     if abs(ped_loc[0]) < 3.5 and abs(ped_loc[1]) < 1.4:
            #         return (True, True, target)

            # # if target_waypoint.road_id == ego_vehicle_waypoint.road_id and \
            # #         target_waypoint.lane_type == ego_vehicle_waypoint.lane_type:
                
            #     elif is_within_distance_ahead(target.get_transform(), self.ego_vehicle.get_transform(), 8.0):
            #         return (False, True, target)

        return (False, False, None)


    def distance(self, source_transform, destination_transform):
        dx = source_transform.location.x - destination_transform.location.x
        dy = source_transform.location.y - destination_transform.location.y

        return math.sqrt(dx * dx + dy * dy)


    def get_ego_speed(self):
        ev_speed = get_speed(self.ego_vehicle)
        ev_speed = round(ev_speed, 2)

        if ev_speed <= 0.0:
            ev_speed = 0.0
        
        return ev_speed
