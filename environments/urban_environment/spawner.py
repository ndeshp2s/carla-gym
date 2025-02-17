import sys
import glob
import os
import random
import numpy as np
from math import sqrt
import math
try:
    sys.path.append(glob.glob('/home/niranjan/carla/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla

from environments.urban_environment import carla_config
from environments.urban_environment.walker_spawn_points import *


class Spawner(object):
    def __init__(self):

        self.client = None
        self.world = None
        self.connected_to_server = False


        self.num_of_ped = carla_config.num_of_ped
        self.num_of_veh = carla_config.num_of_veh

        self.pedestrian_list = []

        self.pedestrian_ids = [0 for i in range(carla_config.num_of_ped)]


    def reset(self):
        self.connect_to_server()
        self.pedestrian_list = []
        self.pedestrian_ids = [0 for i in range(carla_config.num_of_ped)]


    def connect_to_server(self):
        try:
            self.client = carla.Client('localhost', 2000)
            self.client.set_timeout(10.0)
            self.world = self.client.get_world()
            self.connected_to_server = True

            self.world.set_pedestrians_cross_factor(carla_config.percentage_pedestrians_crossing)
            self.world.set_pedestrians_cross_illegal_factor(carla_config.percentage_pedestrians_crossing_illegal)

        except (RuntimeError):
            self.client = None
            self.world = None
            self.ev_id = None
            pass


    def run_step(self, ev_trans = None, step_num = 0):

        ev_trans = self.get_ev_trans()

        spawn_points = self.get_spawn_points(ev_trans)

        
        self.controller_turnon(spawn_points)

        self.check_pedestrian_distance(ev_trans)


        self.spawn_pedestrian(spawn_points, ev_trans)

            
        
    def spawn_pedestrian(self, spawn_points, ev_trans = None):

        if self.connected_to_server is False:
            return

        if ev_trans is None:
            return

        if len(self.pedestrian_list) >=  self.num_of_ped:
            return

        # Add pedestrian
        ped_bp = random.choice(self.world.get_blueprint_library().filter("walker.pedestrian.*"))
        ped_id = next((index for index, value in enumerate(self.pedestrian_ids) if value == 0), None)

        ped_bp.set_attribute('role_name', str(ped_id + 1))

        ped = None
        sp = carla.Transform()

        counter = 0
        while len(self.pedestrian_list) <  self.num_of_ped:
            if len(spawn_points) == 0 or counter >= self.num_of_ped:
                return

            # Add pedestrian
        	ped_bp = random.choice(self.world.get_blueprint_library().filter("walker.pedestrian.*"))
        	ped_id = next((index for index, value in enumerate(self.pedestrian_ids) if value == 0), None)

        	ped_bp.set_attribute('role_name', str(ped_id + 1))

        	ped = None
        	sp = carla.Transform()

            sp = random.choice(spawn_points)
            sp.location.x += random.uniform(-0.5, 0.5)
            sp.location.y += random.uniform(-0.5, 0.5)

            ped = self.world.try_spawn_actor(ped_bp, sp)

            if ped is not None:
                self.pedestrian_ids[ped_id] = 1
                self.pedestrian_list.append({"id": ped.id, "controller": None, "start": sp})
                spawn_points.remove(sp)

            counter = counter + 1


    def controller_turnon(self, goal_points):
        controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')

        for p in self.pedestrian_list:
            if p["controller"] is None:
                ped = self.world.get_actor(p["id"])
                controller = self.world.spawn_actor(controller_bp, carla.Transform(), attach_to = ped)
                controller.start()
                goal = self.get_goal(goal_points, p["start"])
                controller.go_to_location(goal.location)
                controller.set_max_speed(round(random.uniform(0.2, 0.4), 2))

                index = self.pedestrian_list.index(p)
                self.pedestrian_list[index]["controller"] = controller.id


    def check_pedestrian_distance(self, ev_trans = None):
        if ev_trans is None:
            return

        for p in self.pedestrian_list:

            ped = self.world.get_actor(p["id"])
            if ped is None:
                continue

            ped_trans = ped.get_transform()

            if not self.is_within_distance(ped_trans.location, ev_trans.location, ev_trans.rotation.yaw, carla_config.ped_max_dist, carla_config.ped_min_dist, spawn = False):    
                if p["controller"] is not None:
                    controller = self.world.get_actor(p["controller"])
                    if controller is not None:
                        controller.stop()
                        controller.destroy()

                ped_id = int(ped.attributes['role_name'])
                self.pedestrian_ids[ped_id - 1] = 0

                ped.destroy()

                self.pedestrian_list.remove(p)
                
                
    def get_ev_trans(self):
        ev = self.get_ev()

        if ev is not None:
            return ev.get_transform()

        return None


    def get_ev(self):
        actors = self.world.get_actors().filter(carla_config.ev_bp)

        if actors is not None:
            for a in actors:
                if a.attributes.get('role_name') == carla_config.ev_name: 
                    return a
        else:
            return None


    def destroy_all(self):

        world = self.client.get_world()
        if world is not None:
            actor_list = world.get_actors()

            for a in actor_list.filter("walker.pedestrian.*"):
                a.destroy()


    def is_within_distance(self, tar_loc, cur_loc, rot, max_dist, min_dist, spawn = True):

        if not spawn:
            if abs(tar_loc.x - cur_loc.x) > carla_config.ped_max_dist or abs(tar_loc.y - cur_loc.y) > carla_config.ped_max_dist:
                return False

        tar_vec = np.array([tar_loc.x - cur_loc.x, tar_loc.y - cur_loc.y])
        norm = np.linalg.norm(tar_vec)

        for_vec = np.array([math.cos(math.radians(rot)), math.sin(math.radians(rot))])
        d_ang = math.degrees(math.acos(np.dot(for_vec, tar_vec) / norm))

        if norm < 0.001:
            return True

        if (norm > max_dist or norm < min_dist):
            return False

        return d_ang < 90.0


    def get_spawn_points(self, ev_trans = None):
        if ev_trans is None:
            return
        spawn_points = []

        for sp in walker_spawn_points:
            if self.is_within_distance(sp.location, ev_trans.location, ev_trans.rotation.yaw, carla_config.ped_spawn_max_dist, carla_config.ped_spawn_min_dist):
                spawn_points.append(sp)

        return spawn_points


    def get_goal(self, goal_points, start):
        for goal in goal_points:
            goal_wp = self.world.get_map().get_waypoint(goal.location, project_to_road = True, lane_type = carla.LaneType.Any)
            start_wp = self.world.get_map().get_waypoint(start.location, project_to_road = True, lane_type = carla.LaneType.Any)
             

            if (goal_wp.lane_id ^ start_wp.lane_id) < 0 and goal_wp.road_id == start_wp.road_id and self.distance(goal, start) < 15.0:
                return goal
            
        return random.choice(walker_goal_points)


    def distance(self, source_transform, destination_transform):
        dx = source_transform.location.x - destination_transform.location.x
        dy = source_transform.location.y - destination_transform.location.y

        return math.sqrt(dx * dx + dy * dy)



# Only for testing
import sys, os
import glob

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla


def destroy(world):
    actor_list = world.get_actors()
    for a in actor_list.filter("walker.pedestrian.*"):
        a.destroy()
    for a in actor_list.filter("vehicle.*"):
                a.destroy()


def main():

    client = carla.Client('localhost', 2000)
    client.set_timeout(2.0)

    world = client.get_world()


    destroy(world)

    world.set_pedestrians_cross_factor(0.0)
    world.set_pedestrians_cross_illegal_factor(0.0)

    # Spawn ego vehicle
    sp = carla.Transform(carla.Location(x = 40.0, y = -1.8, z = 0.1), carla.Rotation(yaw = 180))
    bp = random.choice(world.get_blueprint_library().filter('vehicle.audi.etron'))
    bp.set_attribute('role_name', 'hero')

    veh = world.spawn_actor(bp, sp)

    spawner = Spawner()
    spawner.reset()


    try:
        while True:

            spawner.run_step(ev_trans = veh.get_transform())

    except KeyboardInterrupt:
        spawner.destroy_all()
        veh.destroy()



if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')