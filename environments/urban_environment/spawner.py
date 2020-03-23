import sys
import glob
import os
import random
import numpy as np
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


    def run_step(self, ev_trans = None):

        ev_trans = self.get_ev_trans()

        self.controller_turnon()

        #while len(self.pedestrian_list) <  self.num_of_ped:
        self.spawn_pedestrian(ev_trans)
        
        self.check_pedestrian_distance(ev_trans)

    def spawn_pedestrian(self, ev_trans = None):

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
        spawn_point = carla.Transform()

        n = 0
        while n < 100:
            spawn_point = random.choice(walker_spawn_points)
            loc = spawn_point.location
            #loc = carla.Location(x = 18.0, y = 5.0, z = 1)
            if self.is_within_distance(loc, ev_trans.location, ev_trans.rotation.yaw, carla_config.ped_spawn_max_dist, carla_config.ped_spawn_min_dist):
                spawn_point.location = loc
                ped = self.world.try_spawn_actor(ped_bp, spawn_point)
                break

            n  = n + 1

        if ped is not None:
            self.pedestrian_ids[ped_id] = 1
            self.pedestrian_list.append({"id": ped.id, "controller": None})


    def controller_turnon(self):

        controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')

        for p in self.pedestrian_list:
            if p["controller"] is None:
                ped = self.world.get_actor(p["id"])
                controller = self.world.spawn_actor(controller_bp, carla.Transform(), attach_to = ped)
                controller.start()
                controller.go_to_location(self.world.get_random_location_from_navigation())
                #controller.go_to_location(carla.Location(x = 18.0, y = -5.2, z = 0.3))
                controller.set_max_speed(float(1.0))

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

            tar_vec = np.array([ev_trans.location.x -ped_trans.location.x, ev_trans.location.y - ped_trans.location.y])
            norm = np.linalg.norm(tar_vec)

            rot = ev_trans.rotation.yaw

            for_vec = np.array([math.cos(math.radians(rot)), math.sin(math.radians(rot))])
            d_ang = math.degrees(math.acos(np.dot(for_vec, tar_vec) / norm))

            #if norm > 50.0 or norm < -5.0:
            if not self.is_within_distance(ped_trans.location, ev_trans.location, ev_trans.rotation.yaw, carla_config.ped_max_dist, carla_config.ped_min_dist):
                
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


        # if sp is None:

        #     world = self.client.get_world()

        #     walker_bp = random.choice(self.blueprints_walkers)

        #     loc = world.get_random_location_from_navigation()

        #     while not self.is_within(loc, ev_trans.location, ev_trans.rotation.yaw, 60.0, 20.0):
        #         loc = world.get_random_location_from_navigation()

        #     #spawn walker
        #     walker_bp = random.choice(world.get_blueprint_library().filter("walker.pedestrian.*"))
        #     controller_bp = world.get_blueprint_library().find('controller.ai.walker')

        #     print(walker_bp)
        #     print(loc)
        #     spawn_point = carla.Transform()
        #     spawn_point.location = loc

        #     walker = world.try_spawn_actor(walker_bp, spawn_point)

        #     if walker is not None:
        #         controller = world.spawn_actor(controller_bp, carla.Transform(), attach_to=walker)

        #         controller.start()



            #print(walker_bp)
            #print(loc)


    def destroy_all(self):
        world = self.client.get_world()
        if world is not None:
            actor_list = world.get_actors()

            for a in actor_list.filter("walker.pedestrian.*"):
                a.destroy()




    def is_within_distance(self, tar_loc, cur_loc, rot, max_dist, min_dist, spawn = True):

        tar_vec = np.array([tar_loc.x - cur_loc.x, tar_loc.y - cur_loc.y])
        norm = np.linalg.norm(tar_vec)

        for_vec = np.array([math.cos(math.radians(rot)), math.sin(math.radians(rot))])
        d_ang = math.degrees(math.acos(np.dot(for_vec, tar_vec) / norm))

        if norm < 0.001:
            return True

        if (norm > max_dist or norm < min_dist):
            return False

        for_vec = np.array([math.cos(math.radians(rot)), math.sin(math.radians(rot))])
        d_ang = math.degrees(math.acos(np.dot(for_vec, tar_vec) / norm))

        return d_ang < 50.0





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
    sp = carla.Transform(carla.Location(x = 220.0, y =129.5, z = 0.1), carla.Rotation(yaw = 180))
    bp = random.choice(world.get_blueprint_library().filter('vehicle.audi.etron'))
    bp.set_attribute('role_name', 'hero')

    veh = world.spawn_actor(bp, sp)

    spawner = Spawner(client)


    try:
        while True:

            spawner.run_step(veh.get_transform())

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