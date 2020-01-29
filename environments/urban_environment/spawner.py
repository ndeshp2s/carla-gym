import random
import numpy as np
import math

class Spawner(object):
    def __init__(self, client = None):

        if client is not None:
            self.client = client
            self.blueprints_walkers = self.client.get_world().get_blueprint_library().filter('walker.pedestrian.*')

        # self.no_of_ped = no_of_ped
        # self.no_of_veh = no_of_veh

        self.walkers_list = []

    def run_step(self, ev_trans = None):

        self.spawn_pedestrian(ev_trans)
        None

    def spawn_pedestrian(self, ev_trans, sp = None, ):

        if sp is None:

            world = self.client.get_world()

            walker_bp = random.choice(self.blueprints_walkers)

            loc = world.get_random_location_from_navigation()

            while not self.is_within(loc, ev_trans.location, ev_trans.rotation.yaw, 60.0, 20.0):
                loc = world.get_random_location_from_navigation()

            #spawn walker
            walker_bp = random.choice(world.get_blueprint_library().filter("walker.pedestrian.*"))
            controller_bp = world.get_blueprint_library().find('controller.ai.walker')

            print(walker_bp)
            print(loc)
            spawn_point = carla.Transform()
            spawn_point.location = loc

            walker = world.try_spawn_actor(walker_bp, spawn_point)

            if walker is not None:
                controller = world.spawn_actor(controller_bp, carla.Transform(), attach_to=walker)

                controller.start()



            #print(walker_bp)
            #print(loc)


    def destroy_all(self):
        world = self.client.get_world()
        if world is not None:
            actor_list = world.get_actors()

            for a in actor_list.filter("walker.pedestrian.*"):
                a.destroy()


    def is_within(self, tar_loc, cur_loc, rot, max_dist, min_dist):

        tar_vec = np.array([tar_loc.x - cur_loc.x, tar_loc.y - cur_loc.y])
        norm = np.linalg.norm(tar_vec)

        if norm < 0.001:
            return True

        if (norm > max_dist or norm < min_dist) or (norm < 10 and norm > 0):
            return False

        for_vec = np.array([math.cos(math.radians(rot)), math.sin(math.radians(rot))])
        d_ang = math.degrees(math.acos(np.dot(for_vec, tar_vec) / norm))

        return d_ang < 90.0







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