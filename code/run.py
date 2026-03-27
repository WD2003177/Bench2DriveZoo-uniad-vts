import argparse
import json
import logging
import math
import sys
import time
import warnings

import libMulticastNetwork
import numpy as np
import vts_map
from chassis.proto.chassis_enums_pb2 import VEHICLE_CONTROL, VEHICLE_FEEDBACK
from chassis.proto.chassis_messages_pb2 import VehicleControl, VehicleFeedback
from get_ip import get_ip_address
from main.proto.enums_pb2 import (
    MT_ACTOR_PREPARE,
    MT_ACTOR_PREPARE_RESULT,
    MT_NOTIFY,
    NT_ABORT_TEST,
    NT_DESTROY_ROLE,
    NT_FINISH_TEST,
    NT_START_TEST,
)
from main.proto.messages_pb2 import ActorPrepare, ActorPrepareResult, Notify
from vts_global_route_planner import get_global_route_planner
from utils import get_logger

warnings.filterwarnings("ignore")
from uniad_b2d_agent import UniadAgent


logger = get_logger(__file__, level=logging.DEBUG)


def prepare():
    logger.info("send prepare result")
    send_prepare_result = ActorPrepareResult()
    send_prepare_result.session_id = session_id
    send_prepare_result.actor_id = actor_id
    send_prepare_result.result = True
    data = send_prepare_result.SerializeToString()
    length = len(data)
    ret = prepare_channel.put(MT_ACTOR_PREPARE_RESULT, length, data)
    if ret != 0:
        logger.error("send prepare msg error")


def get_prepare():
    global recv_prepare
    global session_id
    global actor_id
    global role_id

    ret, msg = prepare_channel.get()
    if msg is None:
        return
    if ret >= 0 and msg.type() == MT_ACTOR_PREPARE:
        recv_prepare = True
        data = libMulticastNetwork.getMessageData(msg)
        prepare_msg = ActorPrepare()
        prepare_msg.ParseFromString(data)
        session_id = prepare_msg.session_id

        brief_data = json.loads(prepare_msg.archive_info.brief_data)
        global_route_planner.change_map(brief_data["zjl_odv_file"])
        target_state = brief_data["testees"][0]["target_state"]
        init_state = brief_data["testees"][0]["init_state"]
        role_id = brief_data["testees"][0]["role_id"]

        init_xyz = vts_map.XYZ(init_state["x"], init_state["y"], init_state["z"])
        target_xyz = vts_map.XYZ(target_state["x"], target_state["y"], target_state["z"])
        logger.debug(f"init xyz: {init_state['x']}, {init_state['y']}, {init_state['z']}")
        logger.debug(f"target xyz: {target_state['x']}, {target_state['y']}, {target_state['z']}")
        route = global_route_planner.trace_route(init_xyz, target_xyz)
        agent.init(route)
        logger.info(f"prepare received, session id: {session_id}, role id: {role_id}")



def process_image_msg(images):
    ret = []
    for i, image in enumerate(images):
        # print(image.timestamp_sec, image.width, image.height)
        # print(image.data.shape)
        img = image.data.astype(np.uint8).reshape(900, 1600, 3)
        ret.append(img)
    return ret


def get_image():
    t0 = time.time()
    msg = image_channel.get_image()
    t1 = time.time()
    if len(msg) == 0:
        return None
    logger.debug("get image time: {}".format(t1 - t0))
    return process_image_msg(msg)


def process_notify():
    global start_test
    global recv_prepare
    ret, msg = notify_channel.get()
    if msg is None:
        return
    if ret >= 0 and msg.type() == MT_NOTIFY:
        notify = Notify()
        data = libMulticastNetwork.getMessageData(msg)
        notify.ParseFromString(data)
        if notify.type == NT_ABORT_TEST or notify.type == NT_FINISH_TEST:
            logger.info("finish session")
            start_test = False
            recv_prepare = False
            agent.reset()
        elif notify.type == NT_START_TEST:
            logger.info("start session")
            start_test = True
        elif notify.type == NT_DESTROY_ROLE:
            pass
        else:
            logger.debug(
                "session id: {}, event type: {}, time:{}".format(
                    notify.session_id, notify.type, notify.header.sim_ts
                )
            )


def send_control_cmd(cmd: VehicleControl):
    data = cmd.SerializeToString()
    length = len(data)
    ret = cmd_channel.put(VEHICLE_CONTROL, length, data)
    if ret == 0:
        logger.debug(
            "[PUT] Channel: {} | Send VEHICLE_CONTROL SUCCESS | ret: {} | length: {} bytes | acc: {:.2f}, speed: {:.2f}, steer: {:.2f}".format(
                cmd_channel.name(), ret, length, cmd.acceleration, cmd.speed, cmd.steering_control.target_steering_wheel_angle
            )
        )
    else:
        logger.error(
            "[PUT] Channel: {} | Send VEHICLE_CONTROL FAILED | ret: {}".format(
                cmd_channel.name(), ret
            )
        )

def get_vehicle_feedback():
    for _ in range(100):  
        ret, msg = cmd_channel.get()
        if msg is None or ret < 0:
            break

        if msg.type() == VEHICLE_FEEDBACK:
            _ = libMulticastNetwork.getMessageData(msg)

def get_vehicle_pose():
    ins = ins_channel.get_ins()
    if ins.sequence_num == 0:
        return None
    return ins


def abort_test():
    global session_id
    global role_id
    notify = Notify()
    notify.header.send_ts = int(time.time() * 1000)
    notify.session_id = session_id
    notify.role_id = role_id
    notify.type = NT_ABORT_TEST
    data = notify.SerializeToString()
    length = len(data)
    notify_channel.put(MT_NOTIFY, length, data)


def tick():
    tick_data = {}
    tick_data['timestamp'] = time.time()
    images = get_image()
    if images is None or len(images) != 6:
        return None
    cams = ['CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT','CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT']
    images = {cam: images[i] for i, cam in enumerate(cams)}

    ins = get_vehicle_pose()
    x = ins.position.x
    y = ins.position.y

    theta = ins.heading

    vx = ins.linear_velocity.x
    vy = ins.linear_velocity.y
    vz = ins.linear_velocity.z
    speed = math.sqrt(vx * vx + vy * vy + vz * vz)

    ax = ins.linear_acceleration.x
    ay = ins.linear_acceleration.y
    az = ins.linear_acceleration.z
    
    tick_data = {**tick_data, **images}
    tick_data['SPEED'] = np.array([speed])
    tick_data['IMU'] = np.array([ax, ay, az, ins.angular_velocity.x, ins.angular_velocity.y, ins.angular_velocity.z, theta])
    tick_data['POS'] = np.array([x, y])
    logger.debug(f"tick data speed: {speed}, pos: {x},{y}, theta: {theta}")
    logger.debug(f"tick data acc: {ax}, {ay}, {az}, ang_vel: {ins.angular_velocity.x}, {ins.angular_velocity.y}, {ins.angular_velocity.z}")
    return tick_data


def main():
    while 1:
        process_notify()
        if not recv_prepare:
            get_prepare()
            time.sleep(0.1)
            continue
        if recv_prepare and not start_test:
            time.sleep(5)
            prepare()
            time.sleep(5)
            continue

        tick_data = tick()
        if tick_data is not None:
            cmd = agent.run_step(tick_data)
            send_control_cmd(cmd)
        else:
            logger.error("[ERROR] tick_data is None! ")
            time.sleep(0.1)
        get_vehicle_feedback()


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--config_center", type=str, default="www.zjvts.cn:52009"
    )
    arg_parser.add_argument("--field_id", type=str, default="unique_fieldid")
    arg_parser.add_argument("--net_interface", type=str, default="eno2")
    args = arg_parser.parse_args()
    param = libMulticastNetwork.CreateChannelsParam()

    local_ip = get_ip_address(args.net_interface)

    #######################################################
    ###################### 需要修改 ########################
    param.config_center_addr = args.config_center  # 组播配置中心的ip
    param.local_ip = "192.168.1.207"  # 本机ip
    param.net_interface_name = args.net_interface  # 本机网卡
    param.field_id = (
        args.field_id
    )  # 唯一的场地id，可以任意字符串，需要和daemon和simulator一致
    #######################################################

    param.log_level = 1  # 1-info, 2-warning, 3-error， 设置不同的组播日志等级
    param.client_name = "apollo_testee"
    param.recv_self_msg = False
    session_id = ""

    channels = libMulticastNetwork.ChannelPtrVector()
    ret = libMulticastNetwork.create_channels(param, channels)
    if ret:
        logger.error("create channels failed, ret: {}".format(ret))
        sys.exit(1)
    channel_map = {}
    # 不同的组播消息通道，用于接收和发送消息
    for c in channels:
        logger.info("message channel name: {}, id: {}".format(c.name(), c.id()))
        channel_map[c.name()] = c

    pointcloud_channel = channel_map["lidar"]
    notify_channel = channel_map["notify"]
    cmd_channel = channel_map["vehiclecontrol"]
    prepare_channel = channel_map["prepare"]
    ins_channel = channel_map["ins"]
    image_channel = channel_map["camera"]

    if not libMulticastNetwork.InitImageDecoder(6, 1600, 900):
        logger.error("image decoder init error")
        sys.exit(1)

    recv_prepare = False
    start_test = False
    actor_id = "apollo_testee"
    role_id = "apollo_testee"

    global_route_planner = get_global_route_planner()
    conf_file = "Bench2DriveZoo/adzoo/uniad/configs/stage2_e2e/tiny_e2e_b2d.py+Bench2DriveZoo/ckpts/uniad_tiny_b2d.pth"
    agent = UniadAgent()
    agent.setup(conf_file)

    main()
