import carla
from agents.navigation.local_planner import LocalPlanner
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
import agents

class Planner(object):
    def __init__(self, vehicle = None):

        self._vehicle = None
        self.global_planner = None
        self.local_planner = None
        self.resolution = 20.0
        self.route_trace = None

        if vehicle is not None:
            self.initialize(vehicle)

    def initialize(self, vehicle):
        self._vehicle = vehicle
        dao = GlobalRoutePlannerDAO(self._vehicle.get_world().get_map(), self.resolution)
        self.global_planner = GlobalRoutePlanner(dao)
        self.global_planner.setup()
        self.local_planner = LocalPlanner(self._vehicle)



    def set_destination(self, location):

        start_waypoint = self._vehicle.get_world().get_map().get_waypoint(self._vehicle.get_location())
        end_waypoint = self._vehicle.get_world().get_map().get_waypoint(carla.Location(location[0], location[1], location[2]))

        self.route_trace = self.global_planner.trace_route(start_waypoint.transform.location, end_waypoint.transform.location)

        #self.route_trace.pop(0)
        #assert self.route_trace

        self.local_planner.set_global_plan(self.route_trace)

    def run_step(self):
        return self.local_planner.run_step(False)

    def view_plan(self):
        for w in self.route_trace:
            self._vehicle.get_world().debug.draw_string(w[0].transform.location, 'o', draw_shadow=False,
                                       color=carla.Color(r=255, g=0, b=0), life_time=120.0,
                                       persistent_lines=True)

    def done(self):
        return self.local_planner.done()

