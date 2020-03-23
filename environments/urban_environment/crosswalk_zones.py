DEBUG = 0


import os
import sys
import glob
try:
    sys.path.append(glob.glob('/home/niranjan/carla/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
    
import carla

if DEBUG:
    # Connect to client
    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(2.0)
    world = client.get_world()


crosswalks = []
corner = carla.Location()

## Junction 1
# crosswalk 1
zone = []
corner = carla.Location(x = 7.4, y = -4.6, z = 0.1)
zone.append(corner)
corner = carla.Location(x = 7.4, y = 4.6, z = 0.1)
zone.append(corner)
corner = carla.Location(x = 4.4, y = -4.6, z = 0.1)
zone.append(corner)
corner = carla.Location(x = 4.4, y = 4.6, z = 0.1)
zone.append(corner)
crosswalks.append(zone)

# crosswalk 2
zone = []
corner = carla.Location(x = 5.0, y = -4.0, z = 0.1)
zone.append(corner)
corner = carla.Location(x = -4.0, y = -4.0, z = 0.1)
zone.append(corner)
corner = carla.Location(x = -4.0, y = -6.9, z = 0.1)
zone.append(corner)
corner = carla.Location(x = 5.0, y = -6.9, z = 0.1)
zone.append(corner)
crosswalks.append(zone)

# crosswalk 3
zone = []
corner = carla.Location(x = -6.8, y = -4.6, z = 0.1)
zone.append(corner)
corner = carla.Location(x = -6.8, y = 4.6, z = 0.1)
zone.append(corner)
corner = carla.Location(x = -3.8, y = -4.6, z = 0.1)
zone.append(corner)
corner = carla.Location(x = -3.8, y = 4.6, z = 0.1)
zone.append(corner)
crosswalks.append(zone)

# crosswalk 4
zone = []
corner = carla.Location(x = 5.0, y = 4.0, z = 0.1)
zone.append(corner)
corner = carla.Location(x = -4.0, y = 4.0, z = 0.1)
zone.append(corner)
corner = carla.Location(x = -4.0, y = 7.1, z = 0.1)
zone.append(corner)
corner = carla.Location(x = 5.0, y = 7.1, z = 0.1)
zone.append(corner)
crosswalks.append(zone)



## Junction 2
# crosswalk 1
zone = []
corner = carla.Location(x = 7.4, y = 45.2, z = 0.1)
zone.append(corner)
corner = carla.Location(x = 7.4, y = 54.5, z = 0.1)
zone.append(corner)
corner = carla.Location(x = 4.4, y = 54.5, z = 0.1)
zone.append(corner)
corner = carla.Location(x = 4.4, y = 45.2, z = 0.1)
zone.append(corner)
crosswalks.append(zone)

# # crosswalk 2
zone = []
corner = carla.Location(x = 5.0, y = 45.8, z = 0.1)
zone.append(corner)
corner = carla.Location(x = -4.0, y = 45.8, z = 0.1)
zone.append(corner)
corner = carla.Location(x = -4.0, y = 42.8, z = 0.1)
zone.append(corner)
corner = carla.Location(x = 5.0, y = 42.8, z = 0.1)
zone.append(corner)
crosswalks.append(zone)

# # crosswalk 3
zone = []
corner = carla.Location(x = -6.8, y = 45.2, z = 0.1)
zone.append(corner)
corner = carla.Location(x = -6.8, y = 54.5, z = 0.1)
zone.append(corner)
corner = carla.Location(x = -3.8, y = 54.5, z = 0.1)
zone.append(corner)
corner = carla.Location(x = -3.8, y = 45.2, z = 0.1)
zone.append(corner)
crosswalks.append(zone)

# # crosswalk 4
zone = []
corner = carla.Location(x = 5.0, y = 53.9, z = 0.1)
zone.append(corner)
corner = carla.Location(x = -4.0, y = 53.9, z = 0.1)
zone.append(corner)
corner = carla.Location(x = -4.0, y = 56.9, z = 0.1)
zone.append(corner)
corner = carla.Location(x = 5.0, y = 56.9, z = 0.1)
zone.append(corner)
crosswalks.append(zone)



# crosswalk
zone = []
corner = carla.Location(x = -126.5, y = 53.9, z = 0.1)
zone.append(corner)
corner = carla.Location(x = -129.5, y = 53.9, z = 0.1)
zone.append(corner)
corner = carla.Location(x = -126.5, y = 45.7, z = 0.1)
zone.append(corner)
corner = carla.Location(x = -129.5, y = 45.7, z = 0.1)
zone.append(corner)

crosswalks.append(zone)


# Display zones
if DEBUG:
    print(len(crosswalks))
    for z in crosswalks:
        print(len(z))
        for c in z:
            world.debug.draw_string(c, 'O', draw_shadow = False, color = carla.Color(r = 255, g = 0, b = 0), life_time = 10.0, persistent_lines = True)


from shapely.geometry import Point, Polygon

def within_crosswalk(x, y):
    within = False
    point = Point(x, y)

    coords = []
    for zones in crosswalks:
        coords = []
        for corner in zones:
            coords.append((corner.x, corner.y))
        
        poly = Polygon(coords)
        
        if point.within(poly):
            within =  True

        coords.clear()

    return within

if DEBUG:
    p = carla.Location(x=-3.5, y=4.1, z = 0.0)
    world.debug.draw_string(p, 'O', draw_shadow = False, color = carla.Color(r = 0, g = 0, b = 255), life_time = 10.0, persistent_lines = True)
    print(within_crosswalk(p.x, p.y))


