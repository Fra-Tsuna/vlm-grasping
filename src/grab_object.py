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
    for name, marker in zip(name_list.markers, centroid_list.markers):
        if name.text == ' soda':
            cup = copy.deepcopy(marker)

    cup_array = np.array([cup.pose.position.x, cup.pose.position.y, cup.pose.position.z])
    cup_array = np.dot(np.transpose(R_m2b), cup_array-T_m2b)
    cup.pose.position.x = cup_array[0]
    cup.pose.position.y = cup_array[1]
    cup.pose.position.z = cup_array[2]

    # px = maximum.pose.position.x
    # py = maximum.pose.position.y
    # pz = maximum.pose.position.z
    # p = np.array([px, py, pz])
    # p = np.dot(np.transpose(R_m2b), p-T_m2b) 
    # px, py, pz = p[0], p[1], p[2]
    # print p

    # # p_arm = np.array([px, py, pz]) - T_x2a1 #segmento distanza tra il punto e il braccio
    # # p_base = np.array([px, py, pz]) - T_m2b
    # # p_base_rot = np.dot(np.transpose(R_m2b), p_base)

    # # print(p_base_rot)
   

    moveit_commander.roscpp_initialize(sys.argv) 
    scene = moveit_commander.PlanningSceneInterface()
    robot = moveit_commander.RobotCommander()
    arm_torso_group = moveit_commander.MoveGroupCommander("arm_torso")
    arm_group = moveit_commander.MoveGroupCommander("arm")
    gripper = moveit_commander.MoveGroupCommander("gripper")

    #solleva bicchiere
    down(arm_torso_group, 0.28)
    forward(arm_group, 0.34)
    close_grippers(gripper)
    back(arm_group, 0.34)
    up(arm_torso_group, 0.28)

    #portalo a destra
    right(arm_group, 0.3)
    down(arm_torso_group, 0.23)
    forward(arm_group, 0.42)
    open_grippers(gripper)
    back(arm_group, 0.42)
    up(arm_torso_group, 0.23)
    
    #solleva la fanta
    left(arm_group, 0.18)
    down(arm_torso_group, 0.24)
    forward(arm_group, 0.41)
    close_grippers(gripper)
    back(arm_group, 0.41)
    up(arm_torso_group, 0.24)

    #spostala a sinistra
    left(arm_group, 0.12)
    down(arm_torso_group, 0.2)
    forward(arm_group, 0.35)
    open_grippers(gripper)
    back(arm_group, 0.35)
    up(arm_torso_group, 0.2)

    # close_grippers(gripper)
    # up(arm_torso_group, 0.4)
    # right(arm_torso_group, 0.3)
    # left(arm_torso_group, 0.3)
    # down(arm_torso_group, 0.4)
    # open_grippers(gripper)
    # up(arm_torso_group, 0.4)

    # grab(cup)
    # drop(cup)

    # arm = "arm_torso"
    # gripper = "gripper"
    # arm_group = moveit_commander.MoveGroupCommander(arm)
    # gripper_group = moveit_commander.MoveGroupCommander(gripper)


    # pose = arm_group.get_current_pose().pose
    # print pose
    # fixed_orientation = Quaternion(pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w)

    # pose1 = Pose()
    # pose1.position.x = px
    # pose1.position.y = py
    # pose1.position.z = pz+0.18
    # pose1.orientation = fixed_orientation

    # go_to_pose(arm_group, pose1)
    # go_to_pose(arm_group, pose)

if __name__ == '__main__':
    rospy.init_node('listener', anonymous=True)
    listener()

