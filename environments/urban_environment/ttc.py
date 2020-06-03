import math 
import gym

import sys, os
import glob
import random

#from sympy.geometry import Point, Circle 
from shapely.geometry import Polygon, Point
# from sympy import *
# from sympy.geometry import *

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla


from environments.urban_environment.urban_env import UrbanEnv as CarlaEnv


env = gym.make('Urban-v0')

def intersect(x1, y1, x2, y2, r1, r2):
    dist_sq = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)
    rad_sum_sq = (r1 + r2) * (r1 + r2)

    # print(x1, y1, x2, y2, r1, r2)
    # print(dist_sq, rad_sum_sq)
    dist_sq = round(dist_sq, 2)
    rad_sum_sq = round(rad_sum_sq, 2)
    print(dist_sq, rad_sum_sq)
    print('--------------')

    if (dist_sq <= rad_sum_sq): 
        return 1
    elif (dist_sq > rad_sum_sq): 
        return -1

def find_ttc(ev_trans, ev_speed):
    pedestrian_list = env.world.get_actors().filter('walker.pedestrian.*')

    ev_heading = (ev_trans.rotation.yaw + 360) % 360
    ev_heading = round(ev_heading, 2)
    ev_heading_radians = math.radians(ev_heading)
    ev_heading_radians = round(ev_heading_radians, 2)

    ev_speed = round(ev_speed, 2)
    ev_speed_x = ev_speed*math.cos(ev_heading_radians)
    ev_speed_y = ev_speed*math.sin(ev_heading_radians)
    ev_speed_x = round(ev_speed_x, 2)
    ev_speed_y = round(ev_speed_y, 2)


    dt = 0.0
    ttc = 0.0


    counter = 1
    while dt < 2.0:
        dt = dt + 0.1
        dt = round(dt, 1)

        # ego vehicle
        new_position_x = ev_trans.location.x + ev_speed_x*dt
        new_position_y = ev_trans.location.y + ev_speed_y*dt
        new_position_x = round(new_position_x, 2)
        new_position_y = round(new_position_y, 2)

        # env.world.debug.draw_string(carla.Location(x = new_position_x, y = new_position_y, z = 0.1), 'O', draw_shadow=False, 
        #     color=carla.Color(r=255, g=counter, b=0), life_time=1.0, persistent_lines=True)

        #ev_polygon = Polygon([(new_position_x + 2.0, new_position_y + 2.0), (new_position_x - 2.0, new_position_y + 2.0), (new_position_x - 2.0, new_position_y - 2.0), (new_position_x + 2.0, new_position_y - 2.0)])

        # env.world.debug.draw_string(carla.Location(x = new_position_x + 2, y = new_position_y+2.0, z = 0.1), 'O', draw_shadow=False, 
        #     color=carla.Color(r=0, g=0, b=255), life_time=0.1, persistent_lines=True)
        # env.world.debug.draw_string(carla.Location(x = new_position_x-2.0, y = new_position_y+2.0, z = 0.1), 'O', draw_shadow=False, 
        #     color=carla.Color(r=0, g=0, b=255), life_time=0.1, persistent_lines=True)
        # env.world.debug.draw_string(carla.Location(x = new_position_x-2.0, y = new_position_y-2.0, z = 0.1), 'O', draw_shadow=False, 
        #     color=carla.Color(r=0, g=0, b=255), life_time=0.1, persistent_lines=True)
        # env.world.debug.draw_string(carla.Location(x = new_position_x+2.0, y = new_position_y-2.0, z = 0.1), 'O', draw_shadow=False, 
        #     color=carla.Color(r=0, g=0, b=255), life_time=0.1, persistent_lines=True)

        veh_bb = env.ego_vehicle.bounding_box
        veh_bb.extent.x = 3.0
        veh_bb.extent.y = 1.5
        vertices = veh_bb.get_world_vertices(carla.Transform(carla.Location(x = new_position_x, y = new_position_y, z = ev_trans.location.z), ev_trans.rotation))
        vertices_c = veh_bb.get_world_vertices(ev_trans)
        # print(vertices[0])
        # print(vertices[2])
        # print(vertices[4])
        # print(vertices[6])

        # print("veh bb:", veh_bb.extent)
        #env.world.debug.draw_box(bb, ev_trans.rotation, 0.05, carla.Color(255,0,0,0),0)
        loc = env.ego_vehicle.get_transform().location
        loc.z = 0.2
        env.world.debug.draw_box(carla.BoundingBox( loc, veh_bb.extent), env.ego_vehicle.get_transform().rotation, 0.1, carla.Color(255,0,0,0),0)

        ev_polygon = Polygon([(vertices[0].x, vertices[0].y), (vertices[2].x, vertices[2].y), (vertices[4].x, vertices[4].y), (vertices[6].x, vertices[6].y)])
        ev_polygon_c = Polygon([(vertices_c[0].x, vertices_c[0].y), (vertices_c[2].x, vertices_c[2].y), (vertices_c[4].x, vertices_c[4].y), (vertices_c[6].x, vertices_c[6].y)])

        for ped in pedestrian_list:
            ped_bb = ped.bounding_box
            ped_bb.extent.x = 1.0
            ped_bb.extent.y = 1.0
            #env.world.debug.draw_box(carla.BoundingBox(ped.get_transform().location, ped_bb.extent), ped.get_transform().rotation, 0.05, carla.Color(0,255,0,0),0)

            ped_trans = ped.get_transform()
            ped_heading = (ped_trans.rotation.yaw + 360) % 360
            ped_heading = round(ped_heading, 2)
            ped_heading_radians = math.radians(ped_heading)
            ped_heading_radians = round(ped_heading_radians, 2)

            ped_speed = env.get_speed(ped)
            ped_speed = round(ped_speed, 2)
            ped_speed_x = ped_speed*math.cos(ped_heading_radians)
            ped_speed_y = ped_speed*math.sin(ped_heading_radians)
            ped_speed_x = round(ped_speed_x, 2)
            ped_speed_y = round(ped_speed_y, 2)


            ped_new_position_x = ped_trans.location.x + ped_speed_x*dt
            ped_new_position_y = ped_trans.location.y + ped_speed_y*dt
            ped_new_position_x = round(ped_new_position_x, 2)
            ped_new_position_y = round(ped_new_position_y, 2)

            ped_point = Point(ped_new_position_x, ped_new_position_y).buffer(1.0)
            ped_point_collision = Point(ped_trans.location.x, ped_trans.location.y).buffer(0.5)

            #env.world.debug.draw_string(carla.Location(x = ped_new_position_x, y = ped_new_position_y, z = 0.1), 'O', draw_shadow=False, 
            #    color=carla.Color(r=255, g=counter, b=0), life_time=1.0, persistent_lines=True)

            loc = carla.Location()
            loc.x = ped_new_position_x
            loc.y = ped_new_position_y
            loc.z = ped_trans.location.z
            trans = carla.Transform()
            trans = ev_trans
            trans.x = new_position_x
            trans.y = new_position_y
            f = veh_bb.contains(loc, trans)
            # print(f)
            # if f == True:
            #     ttc = dt
            #     print("ttc:", ttc)
            #     return ttc


            # flag = veh_bb.contains(world_point = carla.Location(x = ped_new_position_x, y = ped_new_position_y, z = 0.1), transform = ped_trans)
            # print(flag)

            # print(ev_polygon, ped_point)
            print('------------------')
            #print(ev_polygon.contains(ped_point))
            print(ev_polygon.intersects(ped_point))

            if ev_polygon_c.intersects(ped_point_collision):
                print('COLLISION')

            #print(intersection(ev_polygon, ped_point))

            #if intersect(x1 = new_position_x, y1 = new_position_y, x2 = ped_new_position_x, y2 = ped_new_position_y, r1 = 2.0, r2 = 2.0) == 1:
            if ev_polygon.intersects(ped_point) == True:
                # env.world.debug.draw_string(carla.Location(x = ped_new_position_x, y = ped_new_position_y, z = 0.1), 'X', draw_shadow=False, 
                # color=carla.Color(r=255, g=255, b=255), life_time=1.0, persistent_lines=True)

                # env.world.debug.draw_box(carla.BoundingBox(carla.Location(x = new_position_x, y= new_position_y, z = 0.5), veh_bb.extent), env.ego_vehicle.get_transform().rotation, 0.05, carla.Color(255,255,255,0),0)
                ttc = dt
                print("ttc:", ttc)
                return ttc

    #print(dt)
        counter = counter + 5
    return 0



        


def spawn_pedestrian():
    # Add pedestrian
    ped_bp = random.choice(env.world.get_blueprint_library().filter("walker.pedestrian.*"))
    sp = carla.Transform(carla.Location(x = 15.0, y = 6.0, z = 0.2))
    ped = env.world.try_spawn_actor(ped_bp, sp)
    return ped



def controller_turnon(ped):
    controller_bp = env.world.get_blueprint_library().find('controller.ai.walker')
    controller = env.world.spawn_actor(controller_bp, carla.Transform(), attach_to = ped)
    controller.start()
    controller.go_to_location(carla.Location(x = 15.0, y = -6.0, z = 0.2))
    controller.set_max_speed(2.0)


def main():

    

    env.reset(client_only = True)

    env.world.set_pedestrians_cross_factor(0.0)
    env. world.set_pedestrians_cross_illegal_factor(1.0)

    # client = carla.Client('localhost', 2000)
    # client.set_timeout(2.0)

    # world = client.get_world()


    # destroy(world)

    # world.set_pedestrians_cross_factor(0.0)
    # world.set_pedestrians_cross_illegal_factor(0.0)

    # # Spawn ego vehicle
    # sp = carla.Transform(carla.Location(x = 28.4, y = -0.40, z = 0.1), carla.Rotation(yaw = 180))
    # bp = random.choice(world.get_blueprint_library().filter('vehicle.audi.etron'))
    # bp.set_attribute('role_name', 'hero')

    # veh = world.spawn_actor(bp, sp)

    # spawner = Spawner()
    # spawner.reset()


    try:
        #ped = spawn_pedestrian()
        env.step(0)
        #controller_turnon(ped)
        steps = 0
        while True:
            action = input('Enter to continue: ')
            action = int(action)
            #action = 0
            print('Steps:', steps)

            #env.step(action)
            for i in range(1):
                env.step(action)

            ev_trans = env.ego_vehicle.get_transform()
            ev_speed = env.get_ego_speed()
            print('Target speed:', env.planner.local_planner.get_target_speed())
            print('ev_speed: ', ev_speed)
            #print('speed: ', env.ego_vehicle.get_velocity())

            ttc = 0
            #ttc = find_ttc(ev_trans, ev_speed)
            #print('Found collision after: ', ttc)

            

            #

            # find_ttc(veh.get_transform())

            # spawner.run_step(ev_trans = veh.get_transform())

            # for p in spawner.pedestrian_list:

            #     ped = world.get_actor(p["id"])
            #     if ped is None:
            #         continue

            #     ped_trans = ped.get_transform()

            #     ped_loc = pedestrian_relative_position(ped_trans = ped_trans, ev_trans = veh.get_transform())
            #     print(ped_loc)

            #     if abs(ped_loc[0]) < 3.5 and abs(ped_loc[1]) < 1.4:
            #         print('collision')

            steps = steps + 1


    except KeyboardInterrupt:
        #spawner.destroy_all()
        #veh.destroy()
        #env.close()
        env.ego_vehicle.destroy()



if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        env.ego_vehicle.destroy()
        pass
    finally:
        env.ego_vehicle.destroy()
        print('\ndone.')