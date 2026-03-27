import math
from enum import Enum

import vts_map
from utils import get_logger

logger = get_logger(__file__)


# --- 1. 基础定义 ---
class RoadOption(Enum):
    VOID = 0
    LANEFOLLOW = 4
    STRAIGHT = 3
    RIGHT = 2
    LEFT = 1
    CHANGELANELEFT = 5
    CHANGELANERIGHT = 6
    ROADEND = 7


class QuinticPolynomial:
    """
    五次多项式求解器：l(s)
    用于计算从 s=0 到 s=end_s 过程中，横向偏移 l 的变化
    """
    def __init__(self, start_l, start_dl, start_ddl, end_l, end_dl, end_ddl, length):
        self.a0 = start_l
        self.a1 = start_dl
        self.a2 = start_ddl / 2.0

        s = length
        s2 = s * s
        s3 = s * s * s
        s4 = s * s * s * s
        s5 = s * s * s * s * s

        h = end_l - start_l
        v1 = start_dl
        v2 = end_dl
        a1 = start_ddl
        a2 = end_ddl

        # 计算系数 a3, a4, a5
        self.a3 = (10*h - 4*v2*s + 0.5*a2*s2 - 6*v1*s - 1.5*a1*s2) / s3
        self.a4 = (-15*h + 7*v2*s - a2*s2 + 8*v1*s + 1.5*a1*s2) / s4
        self.a5 = (6*h - 3*v2*s + 0.5*a2*s2 - 3*v1*s - 0.5*a1*s2) / s5

    def calc_point(self, s):
        """计算 s 处的 l 值"""
        return self.a0 + self.a1*s + self.a2*s**2 + self.a3*s**3 + self.a4*s**4 + self.a5*s**5

    def calc_first_derivative(self, s):
        """计算 s 处的 dl/ds (类似横向速度/偏航角修正)"""
        return self.a1 + 2*self.a2*s + 3*self.a3*s**2 + 4*self.a4*s**3 + 5*self.a5*s**4

def normalize_angle(angle):
    while angle > math.pi: angle -= 2 * math.pi
    while angle <= -math.pi: angle += 2 * math.pi
    return angle


def generate_advanced_trajectory(route, map_api, sampling_distance=1.0):
    """
    path_segments: list of LaneSegment
    map_api: 提供 get_lane_waypoints(seg), get_lane_width(seg) 等方法
    """
    final_trajectory = []
    LANE_CHANGE_LENGTH = 5.0  # 变道长度 (米)
    path_segments = route.lane_ids
    route_start_s = route.begin.s
    route_end_s = route.end.s
    last_wp = None
    logger.info(f"Generating advanced trajectory for {len(path_segments)} segments.")

    for i in range(len(path_segments) - 1):
        curr_seg = path_segments[i]
        next_seg = path_segments[i+1]
        
        # 1. 获取当前路段的原始中心线点 (Ref Line)

        lane_info = map_api.query_lane_info(curr_seg)
        
        # Determine start and end S for this segment
        if i == 0:
            start_s = route_start_s
        else:
            start_s = lane_info.begin
        
        end_s = lane_info.end
        raw_waypoints = vts_map.TracePointVector()
        map_api.calc_lane_center_line_curv(
            curr_seg, start_s, end_s, 
            sampling_distance, raw_waypoints
        )
        logger.info(f"Segment {i}: start_s={start_s}, end_s={end_s}, waypoints={len(raw_waypoints)}")
        if not raw_waypoints: 
            continue

        if last_wp is not None:
            # find closest point to last_wp and trim
            min_dist = float('inf')
            min_idx = 0
            for idx, wp in enumerate(raw_waypoints):
                dist = math.hypot(wp.x - last_wp.x, wp.y - last_wp.y)
                if dist < min_dist:
                    min_dist = dist
                    min_idx = idx
            raw_waypoints = raw_waypoints[min_idx:]

        # 2. 决策 RoadOption
        option = RoadOption.LANEFOLLOW
        
        # 判断逻辑
        is_same_road = (curr_seg.road_id == next_seg.road_id)
        
        if is_same_road:
            # 同路变道逻辑 (假设OpenDRIVE右侧通行)
            # local_id: -1(内) -> -2(外) => Right
            # local_id: 1(内) -> 2(外) => Right (注意正ID方向相反，绝对值变大也是远离中心线)
            if curr_seg.local_id == next_seg.local_id:
                option = RoadOption.LANEFOLLOW
            elif abs(next_seg.local_id) > abs(curr_seg.local_id):
                option = RoadOption.CHANGELANERIGHT
            else:
                option = RoadOption.CHANGELANELEFT
        else:
            # 跨路，判断角度
            if last_wp is not None:
                cur_hdg = last_wp.hdg
            else:
                cur_hdg = raw_waypoints[0].hdg
            next_hdg = raw_waypoints[len(raw_waypoints) // 2].hdg
            ang_diff_deg = cur_hdg - next_hdg
            ang_diff_deg = math.degrees(normalize_angle(ang_diff_deg))
            
            if ang_diff_deg > 20: 
                option = RoadOption.LEFT
            elif ang_diff_deg < -20: 
                option = RoadOption.RIGHT
            else: 
                option = RoadOption.STRAIGHT 

        # 3. 轨迹点生成与平滑
        if option in [RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT]:
            # === 变道处理 (Frenet Frame Smoothing) ===
            
            # 确定变道方向系数 (左正右负，基于Frenet坐标系定义)
            # 在OpenDRIVE中，s随着道路延伸。
            # 这里的 direction 是指相对于当前车道中心线的移动方向。
            # 如果是 ChangeLaneLeft，意味着 l 要增加 (假设左为正); Right 则减小。
            lat_direction = 1.0 if option == RoadOption.CHANGELANELEFT else -1.0
            
            # 获取实际车道宽度作为目标偏移量
            # 注意：应该获取目标车道相对于当前车道的偏移。
            # 简单情况下：target_l = lane_width * direction
            slz = vts_map.SLZ(curr_seg, (lane_info.begin + lane_info.end)/2.0, 0, 0)
            lane_width = map_api.query_lane_width_at(slz) # 假设3.5m
            target_l_offset = lane_width * lat_direction
            
            # 初始化五次多项式
            # start_l=0 (当前车道中心), end_l=target_l_offset
            quintic = QuinticPolynomial(0, 0, 0, target_l_offset, 0, 0, LANE_CHANGE_LENGTH)
            
            current_s_in_change = 0.0
            
            for wp in raw_waypoints:
                # 只在变道距离内进行平滑，超过距离后假定已完成变道(或进入下一段)
                if current_s_in_change <= LANE_CHANGE_LENGTH:
                    
                    # A. 计算当前的横向偏移 l 和 导数 l'
                    l_smooth = quintic.calc_point(current_s_in_change)
                    # l_prime = quintic.calc_first_derivative(current_s_in_change) # dl/ds
                    
                    # B. 投影计算新的 (x, y)
                    # 利用基准点的 hdg 计算法向量方向
                    # 假设 l 正方向为航向角 + 90度 (左手系或右手系需根据具体地图定义确认)
                    # 通用公式：x' = x - l * sin(h), y' = y + l * cos(h) (这是向左为正的推导)
                    new_x = wp.x - l_smooth * math.sin(wp.hdg)
                    new_y = wp.y + l_smooth * math.cos(wp.hdg)
                    
                    # C. 计算新的航向角
                    # 变道导致航向角不仅仅是道路几何航向，还要加上变道切入角
                    # tan(delta_heading) = dl / ds
                    # new_hdg = normalize_angle(wp.hdg + math.atan(l_prime))
                    
                    # D. 构造新的 Waypoint
                    # 继承原点的 z, curv 等，但更新位置和 l 值
                    # new_wp = Waypoint(
                    #     x=new_x, y=new_y, z=wp.z,
                    #     lane_id=curr_seg.local_id, # 变道过程中 id 暂且记为当前或过度
                    #     s=wp.s,
                    #     l=l_smooth, # 记录当前的平滑l值
                    #     hdg=new_hdg,
                    #     curv=wp.curv, # 简化：由于是微小变道，曲率近似沿用
                    #     curv_deriv=wp.curv_deriv
                    # )
                    final_trajectory.append(([new_x, new_y], option))
                    
                    current_s_in_change += sampling_distance
                else:
                    # 超过变道长度，通常意味着已经进入了目标车道的范围
                    # 在简单实现中，我们可以截断，因为 next_seg 的循环会接上
                    break
                last_wp = wp

        else:
            # === 非变道 (Follow, Straight, Turn, etc.) ===
            # 直接使用 API 给的点，不做几何修改
            for wp in raw_waypoints:
                # 确保 wp 的属性是原始干净的
                final_trajectory.append(([wp.x, wp.y], option))

    # --- 处理最后一段 ---
    if len(path_segments) > 0:
        last_seg = path_segments[-1]

        lane_info = map_api.query_lane_info(last_seg)
        
        # Determine start and end S for this segment
        start_s = lane_info.begin
        end_s = route_end_s
        raw_waypoints = vts_map.TracePointVector()
        map_api.calc_lane_center_line_curv(
            last_seg, start_s, end_s, 
            sampling_distance, raw_waypoints
        )
        # trim the last segment waypoints
        if last_wp is not None:
            # find closest point to last_wp and trim
            min_dist = float('inf')
            min_idx = 0
            for idx, wp in enumerate(raw_waypoints):
                dist = math.hypot(wp.x - last_wp.x, wp.y - last_wp.y)
                if dist < min_dist:
                    min_dist = dist
                    min_idx = idx
            raw_waypoints = raw_waypoints[min_idx:]
        for wp in raw_waypoints:
            final_trajectory.append(([wp.x, wp.y], RoadOption.LANEFOLLOW))
            
    return final_trajectory


