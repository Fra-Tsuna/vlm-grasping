#!/usr/bin/env python

import rospy
import open3d as o3d
import os
import numpy as np
import copy
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
import math
from math import pi
import rospy
import sys
from geometry_msgs.msg import Pose, Quaternion
import moveit_commander
import moveit_msgs.msg
import tf2_ros
import tf2_py as tf2
import pickle
from low_level_actions import *

ROOT_DIR = os.path.abspath(__file__+'/../..')
SCAN_DIR = ROOT_DIR+'/images/test_order/'

CONFIG_DIR = ROOT_DIR+'/config/dump_order/'

ARM_REACH = 1.5
OFFSET_GRIPPER = 0.15
rospy.init_node('listener', anonymous=True)
tf_buffer = tf2_ros.Buffer()
tf_listener = tf2_ros.TransformListener(tf_buffer)

# display_trajectory_publisher = rospy.Publisher(
#     "/move_group/display_planned_path",
#     moveit_msgs.msg.DisplayTrajectory,
#     queue_size=1,
# )
# publisher = rospy.Publisher("/pcl_centroids", MarkerArray, queue_size=100)

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
    centroid_list = rospy.wait_for_message("/pcl_centroids", MarkerArray)
    name_list = rospy.wait_for_message("/pcl_names", MarkerArray)

    trans_base = tf_buffer.lookup_transform("map", "base_footprint",  rospy.Time(0), rospy.Duration(2.0))
    R_m2b, T_m2b = get_R_and_T(trans_base)

    cup = None
    can = None
    for name, marker in zip(name_list.markers, centroid_list.markers):
        if name.text == ' soda':
            can = copy.deepcopy(marker)
        if name.text == ' cup':
            cup = copy.deepcopy(marker)

    cup_array = np.array([cup.pose.position.x, cup.pose.position.y, cup.pose.position.z])
    cup_array = np.dot(np.transpose(R_m2b), cup_array-T_m2b)
    cup.pose.position.x = cup_array[0]
    cup.pose.position.y = cup_array[1]
    cup.pose.position.z = cup_array[2]

    can_array = np.array([can.pose.position.x, can.pose.position.y, can.pose.position.z])
    can_array = np.dot(np.transpose(R_m2b), can_array-T_m2b)
    can.pose.position.x = can_array[0]
    can.pose.position.y = can_array[1]
    can.pose.position.z = can_array[2]

    moveit_commander.roscpp_initialize(sys.argv) 
    scene = moveit_commander.PlanningSceneInterface()
    robot = moveit_commander.RobotCommander()
    arm_torso_group = moveit_commander.MoveGroupCommander("arm_torso")
    arm_group = moveit_commander.MoveGroupCommander("arm")
    gripper = moveit_commander.MoveGroupCommander("gripper")

    grab(arm_group, gripper, cup)
    goal_cup_pose = copy.deepcopy(cup)
    goal_cup_pose.pose.position.y -= 0.3
    drop(arm_group, gripper, goal_cup_pose)
    grab(arm_group, gripper, can)
    goal_can_pose = copy.deepcopy(can)
    goal_can_pose.pose.position.y += 0.15
    drop(arm_group, gripper, goal_can_pose)


if __name__ == '__main__':
    rospy.init_node('listener', anonymous=True)
    listener()

