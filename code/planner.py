import logging
import os
import shutil
import time
from collections import deque

import numpy as np
from utils import get_logger

logger = get_logger(__file__, level=logging.DEBUG)

DEBUG = int(os.environ.get('HAS_DISPLAY', 0))
SAVE_PATH = os.environ.get('SAVE_PATH', None)


class Plotter(object):
    def __init__(self, size):
        self.size = size
        self.clear()
        self.title = str(self.size)

    def clear(self):
        from PIL import Image, ImageDraw

        self.img = Image.fromarray(np.zeros((self.size, self.size, 3), dtype=np.uint8))
        self.draw = ImageDraw.Draw(self.img)

    def dot(self, pos, node, color=(255, 255, 255), r=2):
        x, y = 5.5 * (pos - node)
        x += self.size / 2
        y += self.size / 2

        self.draw.ellipse((x-r, y-r, x+r, y+r), color)

    def show(self):
        if not DEBUG:
            return

        import cv2

        cv2.imshow(self.title, cv2.cvtColor(np.array(self.img), cv2.COLOR_BGR2RGB))
        cv2.waitKey(1)
    
    def save(self, filepath):
        self.img.save(filepath)


class RoutePlanner(object):
    def __init__(self, min_distance, max_distance, debug_size=256):
        self.route = deque()
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.debug = Plotter(debug_size)
        self.first_hit = True
  
    def set_route(self, global_plan):
        self.route.clear()
        for pos, cmd in global_plan:
            self.route.append((np.array(pos), cmd))

    def trim_route(self, gps):
        while len(self.route) > 2:
            distance = np.linalg.norm(self.route[0][0] - gps)
            if distance > self.min_distance:
                self.route.popleft()
            else:
                break

    def run_step(self, gps, heading):
        self.debug.clear()

        if len(self.route) == 1:
            return self.route[0]
        
        if self.first_hit:
            self.first_hit = False
            self.trim_route(gps)

        to_pop = 0
        farthest_in_range = -np.inf
        cumulative_distance = 0.0

        for i in range(1, len(self.route)):
            if cumulative_distance > self.max_distance:
                break

            cumulative_distance += np.linalg.norm(self.route[i][0] - self.route[i-1][0])
            distance = np.linalg.norm(self.route[i][0] - gps)

            if distance <= self.min_distance and distance > farthest_in_range:
                farthest_in_range = distance
                to_pop = i

            r = 255 * int(distance > self.min_distance)
            g = 255 * int(self.route[i][1].value == 4)
            b = 255
            self.debug.dot(gps, self.route[i][0], (r, g, b))
        
        logger.info(f"route planner gps: {gps}, heading: {heading}")
        logger.info(f"route planner to pop waypoints: {to_pop}")
        for i in range(to_pop):
            logger.info(f"route planner popping waypoint {i}: {self.route[i][0]}, command: {self.route[i][1]}")
        logger.info(f"route planner farthest in range: {farthest_in_range}")


        for _ in range(to_pop):
            if len(self.route) > 2:
                self.route.popleft()

        self.debug.dot(gps, self.route[0][0], (0, 255, 0))
        self.debug.dot(gps, self.route[1][0], (255, 0, 0))
        self.debug.dot(gps, gps, (0, 0, 255))
        self.debug.show()
        if SAVE_PATH is not None:
            route_path = f"{SAVE_PATH}/route"
            if os.path.exists(route_path):
               shutil.rmtree(route_path)
            os.mkdir(route_path)
            self.debug.save(f"{route_path}/{int(time.time() * 1000)}.png")
        logger.debug(f"route planner remaining waypoints: {len(self.route)}")
        logger.info(f"route planner next waypoint: {self.route[1][0]}, command: {self.route[1][1]}")
        return self.route[1]
    