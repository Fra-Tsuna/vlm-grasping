#!/usr/bin/env python

import rospy
import copy
import os
import numpy as np
from visualization_msgs.msg import Marker
import math
from math import pi
import rospy
import sys
from geometry_msgs.msg import Pose, Quaternion
import moveit_commander
import moveit_msgs.msg
import tf2_ros
import tf2_py as tf2
ROOT_DIR = os.path.abspath(__file__+'/../..')
SCAN_DIR = ROOT_DIR+'/images/test_order/'

CONFIG_DIR = ROOT_DIR+'/config/dump_order/'

ARM_REACH = 1.5
OFFSET_GRIPPER = 0.15
rospy.init_node('listener', anonymous=True)
tf_buffer = tf2_ros.Buffer()
tf_listener = tf2_ros.TransformListener(tf_buffer)

def get_R_and_T(trans):
    Tx_base = trans.transform.translation.x
    Ty_base = trans.transform.translation.y
    Tz_base = trans.transform.translation.z
    T = np.array([Tx_base, Ty_base, Tz_base])
    # Quaternion coordinates
    qx = trans.transform.rotation.x
    qy = trans.transform.rotation.y
    qz = trans.transform.rotation.z
    qw = trans.transform.rotation.w
    
    # Rotation matrix
    R = 2*np.array([[pow(qw,2) + pow(qx,2) - 0.5, qx*qy-qw*qz, qw*qy+qx*qz],[qw*qz+qx*qy, pow(qw,2) + pow(qy,2) - 0.5, qy*qz-qw*qx],[qx*qz-qw*qy, qw*qx+qy*qz, pow(qw,2) + pow(qz,2) - 0.5]])
    return R, T

def listener():

    robot = moveit_commander.RobotCommander()
    scene = moveit_commander.PlanningSceneInterface()
    arm_torso = "arm_torso"
    gripper = "gripper"
    arm_group = moveit_commander.MoveGroupCommander(arm_torso)
    gripper_group = moveit_commander.MoveGroupCommander(gripper)

    init = list(np.array([11, -77, -11, 111, -90, 78, 0])*np.pi/180)
    # wp1 = list(np.array([11, 58, -11, 58, -90, 78, 0])*np.pi/180)
    # wp2 = list(np.array([81, 58, -11, 58, -90, 78, 0])*np.pi/180)
    # wp1 = [0.35]+wp1
    # wp2 = [0.35]+wp2


    new_wp1 = list(np.array([52, -25, -114, 126, 92, 72, 49])*np.pi/180)
    new_wp1 = [0.35]+new_wp1

    arm_group.go(new_wp1, wait=True)
    arm_group.stop()

    open_grip = [0.04, 0.04]
    gripper_group.go(open_grip, wait=True)
    gripper_group.stop()

 
if __name__ == '__main__':
    listener()

