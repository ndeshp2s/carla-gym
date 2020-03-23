import os 
import sys
import subprocess
import glob
import time
from os import path, environ
import psutil
import gym

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla


class CarlaGym(gym.Env):
    def __init__(self):
        super(CarlaGym, self).__init__()
        self.server = None
        self.client = None
        self.world = None
      

    def kill_processes(self):
        binary = 'CarlaUE4.sh'
        for process in psutil.process_iter():


            if process.name().lower().startswith(binary.split('.')[0].lower()):
                print(process)
                try:
                    process.terminate()
                except:
                    pass

        # Check if any are still alive, create a list
        still_alive = []
        for process in psutil.process_iter():
            if process.name().lower().startswith(binary.split('.')[0].lower()):
                still_alive.append(process)

        # Kill process and wait until it's being killed
        if len(still_alive):
            for process in still_alive:
                try:
                    process.kill()
                except:
                    pass
            psutil.wait_procs(still_alive)


    def open_server(self, display = True, rendering = True, town = "Town11", synchronous = True):
        p = None
        cmd = [path.join(environ.get('CARLA_SERVER'), 'CarlaUE4.sh')]

        if not display:
            env_ =  {**os.environ, 'DISPLAY': ''}
            cmd.append(" -opengl")
            p = subprocess.Popen(cmd, env=env_)

        else:
            cmd.append(" -opengl")
            p = subprocess.Popen(cmd)

        
        while True:
            try:
                client = carla.Client('localhost', 2000, 10)

                if client.get_world().get_map().name != town:
                    carla.Client('localhost', 2000, 10).load_world(self.town)
                    while True:
                        try:
                            while carla.Client('localhost', 2000, 10).get_world().get_map().name != town:
                                time.sleep(0.1)
                            break
                        except:
                            pass
                break
            except Exception as e:
                time.sleep(0.1)

        return p


    def setup_client_and_server(self, display = True, rendering = True, reconnect_client_only = False):

        # open the server
        if not reconnect_client_only:
            self.kill_processes()
            self.server = self.open_server(display = display, rendering = rendering)


        # open client
        self.client = carla.Client('localhost', 2000, 10)
        self.world = self.client.get_world()
        


    def apply_settings(self, fps = 10.0, rendering = True):

        self.delta_seconds = 1.0 / fps
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=self.delta_seconds))


    def tick(self):
        self.world.tick()
