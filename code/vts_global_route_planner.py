import os

import vts_map
from planner_utils import generate_advanced_trajectory
from utils import get_logger

logger = get_logger(__file__)


class VtsGlobalRoutePlanner:
    """
    Global Route Planner using vts_map API, providing trace_route functionality.
    """
    def __init__(self, sampling_resolution=2.0):
        self._sampling_resolution = sampling_resolution
        maps = vts_map.get_default_map_file_paths()
        logger.info("default maps: {}".format(maps))
        logger.info("loading map: {}".format(maps[1]))
        self.handle = 0
        self._map = vts_map.Map()
        self._map.load(str(maps[1]), self.handle)
        self._map_file = maps[0]
        self.map_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "maps")
    
    def change_map(self, map_file):
        if map_file == self._map_file:
            return
        self._map.unload(self.handle)
        map_path = os.path.join(self.map_dir, map_file)
        logger.info("changing map to: {}".format(map_path))
        self._map.load(str(map_path), self.handle)
        self._map_file = map_file
    
    def trace_route(self, origin_xyz, destination_xyz):
        """
        This method returns list of (vts_map.TracePoint, RoadOption)
        from origin to destination using vts_map API.
        """
        # Find SLZ for start and end points
        start_slz = vts_map.SLZ()
        self._map.find_slz_global(origin_xyz, start_slz)
        end_slz = vts_map.SLZ()
        self._map.find_slz_global(destination_xyz, end_slz)

        # Create anchors for routing
        start_anchor = vts_map.Anchor()
        start_anchor.id = "start"
        start_anchor.slz = start_slz

        end_anchor = vts_map.Anchor()
        end_anchor.id = "end"
        end_anchor.slz = end_slz

        route = vts_map.Route()
        anchor_array = vts_map.AnchorArray()
        
        # Plan the high-level route using vts_map API
        ret = self._map.plan_route(start_anchor, anchor_array, end_anchor, route)
        if ret != vts_map.ErrorCode.kOK:
            return []
        logger.info("route length: {}".format(route.length))
        for lane in route.lane_ids:
            logger.info("road id: {}, local id: {}, section idx: {}".format(lane.road_id, lane.local_id, lane.section_idx))
        traj = generate_advanced_trajectory(route, self._map, self._sampling_resolution)
        for wp in traj[:10]:
            logger.debug("waypoint: x={}, y={}".format(wp[0][0], wp[0][1]))
        return traj

 
# singleton instance
global_route_planner = None
def get_global_route_planner():
    global global_route_planner
    if global_route_planner is None:
        global_route_planner = VtsGlobalRoutePlanner()
    return global_route_planner