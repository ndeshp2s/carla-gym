import glob
import os
import sys

try:
    sys.path.append(glob.glob('**/*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

import argparse
import random

DEBUG = 0

point_spacing = 10

if DEBUG:
    # Connect to client
    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(2.0)
    world = client.get_world()


# Adding spawn points
walker_spawn_points = []
# spawn_point = carla.Transform(carla.Location(x=10,y=-5,z=0.1), carla.Rotation(yaw=0, pitch=0, roll=0))
# walker_spawn_points.append(spawn_point)


for i in range(15, 90, point_spacing):
  spawn_point = carla.Transform(carla.Location(x=i,y=-105,z=0.5), carla.Rotation(yaw=0, pitch=0, roll=0))
  walker_spawn_points.append(spawn_point)

  spawn_point = carla.Transform(carla.Location(x=i,y=-94.5,z=0.5), carla.Rotation(yaw=0, pitch=0, roll=0))
  walker_spawn_points.append(spawn_point)  

  spawn_point = carla.Transform(carla.Location(x=i,y=-5,z=0.5), carla.Rotation(yaw=0, pitch=0, roll=0))
  walker_spawn_points.append(spawn_point)

  spawn_point = carla.Transform(carla.Location(x=i,y=5.5,z=0.5), carla.Rotation(yaw=0, pitch=0, roll=0))
  walker_spawn_points.append(spawn_point)

  spawn_point = carla.Transform(carla.Location(x=i,y=44.5,z=0.5), carla.Rotation(yaw=0, pitch=0, roll=0))
  walker_spawn_points.append(spawn_point)  

  spawn_point = carla.Transform(carla.Location(x=i,y=55.5,z=0.5), carla.Rotation(yaw=0, pitch=0, roll=0))
  walker_spawn_points.append(spawn_point)  

  spawn_point = carla.Transform(carla.Location(x=i,y=94.5,z=0.5), carla.Rotation(yaw=0, pitch=0, roll=0))
  walker_spawn_points.append(spawn_point)

  spawn_point = carla.Transform(carla.Location(x=i,y=105.0,z=0.5), carla.Rotation(yaw=0, pitch=0, roll=0))
  walker_spawn_points.append(spawn_point)


for i in range(115, 190, point_spacing):
  spawn_point = carla.Transform(carla.Location(x=i,y=-105,z=0.5), carla.Rotation(yaw=0, pitch=0, roll=0))
  walker_spawn_points.append(spawn_point)

  spawn_point = carla.Transform(carla.Location(x=i,y=-94.5,z=0.5), carla.Rotation(yaw=0, pitch=0, roll=0))
  walker_spawn_points.append(spawn_point)  

  spawn_point = carla.Transform(carla.Location(x=i,y=-5,z=0.5), carla.Rotation(yaw=0, pitch=0, roll=0))
  walker_spawn_points.append(spawn_point)

  spawn_point = carla.Transform(carla.Location(x=i,y=5.5,z=0.5), carla.Rotation(yaw=0, pitch=0, roll=0))
  walker_spawn_points.append(spawn_point)

  spawn_point = carla.Transform(carla.Location(x=i,y=44.5,z=0.5), carla.Rotation(yaw=0, pitch=0, roll=0))
  walker_spawn_points.append(spawn_point)  

  spawn_point = carla.Transform(carla.Location(x=i,y=55.5,z=0.5), carla.Rotation(yaw=0, pitch=0, roll=0))
  walker_spawn_points.append(spawn_point)  

  spawn_point = carla.Transform(carla.Location(x=i,y=94.5,z=0.5), carla.Rotation(yaw=0, pitch=0, roll=0))
  walker_spawn_points.append(spawn_point)

  spawn_point = carla.Transform(carla.Location(x=i,y=105.0,z=0.5), carla.Rotation(yaw=0, pitch=0, roll=0))
  walker_spawn_points.append(spawn_point)



for i in range(-185, -15, point_spacing):
  spawn_point = carla.Transform(carla.Location(x=i,y=-105,z=0.5), carla.Rotation(yaw=0, pitch=0, roll=0))
  walker_spawn_points.append(spawn_point)

  spawn_point = carla.Transform(carla.Location(x=i,y=-94.5,z=0.5), carla.Rotation(yaw=0, pitch=0, roll=0))
  walker_spawn_points.append(spawn_point)  

  spawn_point = carla.Transform(carla.Location(x=i,y=-5,z=0.5), carla.Rotation(yaw=0, pitch=0, roll=0))
  walker_spawn_points.append(spawn_point)

  spawn_point = carla.Transform(carla.Location(x=i,y=5.5,z=0.5), carla.Rotation(yaw=0, pitch=0, roll=0))
  walker_spawn_points.append(spawn_point)

  spawn_point = carla.Transform(carla.Location(x=i,y=44.5,z=0.5), carla.Rotation(yaw=0, pitch=0, roll=0))
  walker_spawn_points.append(spawn_point)  

  spawn_point = carla.Transform(carla.Location(x=i,y=55.5,z=0.5), carla.Rotation(yaw=0, pitch=0, roll=0))
  walker_spawn_points.append(spawn_point)  

  spawn_point = carla.Transform(carla.Location(x=i,y=94.5,z=0.5), carla.Rotation(yaw=0, pitch=0, roll=0))
  walker_spawn_points.append(spawn_point)

  spawn_point = carla.Transform(carla.Location(x=i,y=105.0,z=0.5), carla.Rotation(yaw=0, pitch=0, roll=0))
  walker_spawn_points.append(spawn_point)



for i in range(-92, 92, point_spacing):
  spawn_point = carla.Transform(carla.Location(x=-205,y=i,z=0.5), carla.Rotation(yaw=0, pitch=0, roll=0))
  walker_spawn_points.append(spawn_point)  
  spawn_point = carla.Transform(carla.Location(x=206,y=i,z=0.5), carla.Rotation(yaw=0, pitch=0, roll=0))
  walker_spawn_points.append(spawn_point)


for i in range(67, 90, point_spacing):

  spawn_point = carla.Transform(carla.Location(x=195,y=i,z=0.5), carla.Rotation(yaw=0, pitch=0, roll=0))
  walker_spawn_points.append(spawn_point)  
  spawn_point = carla.Transform(carla.Location(x=105.5,y=i,z=0.5), carla.Rotation(yaw=0, pitch=0, roll=0))
  walker_spawn_points.append(spawn_point)  
  spawn_point = carla.Transform(carla.Location(x=94.5,y=i,z=0.5), carla.Rotation(yaw=0, pitch=0, roll=0))
  walker_spawn_points.append(spawn_point)  
  spawn_point = carla.Transform(carla.Location(x=5.5,y=i,z=0.5), carla.Rotation(yaw=0, pitch=0, roll=0))
  walker_spawn_points.append(spawn_point)  
  spawn_point = carla.Transform(carla.Location(x=-5.5,y=i,z=0.5), carla.Rotation(yaw=0, pitch=0, roll=0))
  walker_spawn_points.append(spawn_point)  
  spawn_point = carla.Transform(carla.Location(x=-194.5,y=i,z=0.5), carla.Rotation(yaw=0, pitch=0, roll=0))
  walker_spawn_points.append(spawn_point)

for i in range(20, 43, point_spacing):

  spawn_point = carla.Transform(carla.Location(x=195,y=i,z=0.5), carla.Rotation(yaw=0, pitch=0, roll=0))
  walker_spawn_points.append(spawn_point)  
  spawn_point = carla.Transform(carla.Location(x=105.5,y=i,z=0.5), carla.Rotation(yaw=0, pitch=0, roll=0))
  walker_spawn_points.append(spawn_point)  
  spawn_point = carla.Transform(carla.Location(x=94.5,y=i,z=0.5), carla.Rotation(yaw=0, pitch=0, roll=0))
  walker_spawn_points.append(spawn_point)  
  spawn_point = carla.Transform(carla.Location(x=5.5,y=i,z=0.5), carla.Rotation(yaw=0, pitch=0, roll=0))
  walker_spawn_points.append(spawn_point)  
  spawn_point = carla.Transform(carla.Location(x=-5.5,y=i,z=0.5), carla.Rotation(yaw=0, pitch=0, roll=0))
  walker_spawn_points.append(spawn_point)  
  spawn_point = carla.Transform(carla.Location(x=-194.5,y=i,z=0.5), carla.Rotation(yaw=0, pitch=0, roll=0))
  walker_spawn_points.append(spawn_point)

for i in range(-80, -10, point_spacing):
  spawn_point = carla.Transform(carla.Location(x=195,y=i,z=0.5), carla.Rotation(yaw=0, pitch=0, roll=0))
  walker_spawn_points.append(spawn_point)  
  spawn_point = carla.Transform(carla.Location(x=105.5,y=i,z=0.5), carla.Rotation(yaw=0, pitch=0, roll=0))
  walker_spawn_points.append(spawn_point)  
  spawn_point = carla.Transform(carla.Location(x=94.5,y=i,z=0.5), carla.Rotation(yaw=0, pitch=0, roll=0))
  walker_spawn_points.append(spawn_point)  
  spawn_point = carla.Transform(carla.Location(x=5.5,y=i,z=0.5), carla.Rotation(yaw=0, pitch=0, roll=0))
  walker_spawn_points.append(spawn_point)  
  spawn_point = carla.Transform(carla.Location(x=-5.5,y=i,z=0.5), carla.Rotation(yaw=0, pitch=0, roll=0))
  walker_spawn_points.append(spawn_point)  
  spawn_point = carla.Transform(carla.Location(x=-194.5,y=i,z=0.5), carla.Rotation(yaw=0, pitch=0, roll=0))
  walker_spawn_points.append(spawn_point)

  # spawn_point = carla.Transform(carla.Location(x=i,y=5.5,z=0.5), carla.Rotation(yaw=0, pitch=0, roll=0))
  # walker_spawn_points.append(spawn_point)

  # spawn_point = carla.Transform(carla.Location(x=i,y=44.5,z=0.5), carla.Rotation(yaw=0, pitch=0, roll=0))
  # walker_spawn_points.append(spawn_point)  

  # spawn_point = carla.Transform(carla.Location(x=i,y=55.5,z=0.5), carla.Rotation(yaw=0, pitch=0, roll=0))
  # walker_spawn_points.append(spawn_point)  

  # spawn_point = carla.Transform(carla.Location(x=i,y=94.5,z=0.5), carla.Rotation(yaw=0, pitch=0, roll=0))
  # walker_spawn_points.append(spawn_point)

  # spawn_point = carla.Transform(carla.Location(x=i,y=105.0,z=0.5), carla.Rotation(yaw=0, pitch=0, roll=0))
  # walker_spawn_points.append(spawn_point)



if DEBUG:
    for w in walker_spawn_points:
        world.debug.draw_string(w.location, 'O', draw_shadow=False, 
            color=carla.Color(r=255, g=0, b=0), life_time=200.0,
            persistent_lines=True)
