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
from primitive_actions import *

ROOT_DIR = os.path.abspath(__file__+'/../..')
Strophy_DIR = ROOT_DIR+'/images/test_order/'

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

    middle_shelf = None
    trophy = None
    top_shelf = None
    bottom_shelf = None
    cup = None
    for name, marker in zip(name_list.markers, centroid_list.markers):
        if name.text == 'trophy':
            trophy = copy.deepcopy(marker)
        if name.text == 'middle shelf':
            middle_shelf = copy.deepcopy(marker)
        if name.text == 'top shelf':
            top_shelf = copy.deepcopy(marker)
        if name.text == 'cup':
            cup = copy.deepcopy(marker)
        if name.text == 'bottom shelf':
             bottom_shelf = copy.deepcopy(marker)

    trophy_array = np.array([trophy.pose.position.x, trophy.pose.position.y, trophy.pose.position.z])
    trophy_array = np.dot(np.transpose(R_m2b), trophy_array-T_m2b)
    trophy.pose.position.x = trophy_array[0] + 0.02
    trophy.pose.position.y = trophy_array[1] + 0.07
    trophy.pose.position.z = trophy_array[2] + 0.04

    cup_array = np.array([cup.pose.position.x, cup.pose.position.y, cup.pose.position.z])
    cup_array = np.dot(np.transpose(R_m2b), cup_array-T_m2b)
    cup.pose.position.x = cup_array[0] 
    cup.pose.position.y = cup_array[1] +0.05
    cup.pose.position.z = cup_array[2]  +0.05


    middle_shelf_array = np.array([middle_shelf.pose.position.x, middle_shelf.pose.position.y, middle_shelf.pose.position.z])
    middle_shelf_array = np.dot(np.transpose(R_m2b), middle_shelf_array-T_m2b)
    middle_shelf.pose.position.x = middle_shelf_array[0]
    middle_shelf.pose.position.y = middle_shelf_array[1] + 0.09
    middle_shelf.pose.position.z = middle_shelf_array[2] +0.2

    bottom_shelf_array = np.array([bottom_shelf.pose.position.x, bottom_shelf.pose.position.y, bottom_shelf.pose.position.z])
    bottom_shelf_array = np.dot(np.transpose(R_m2b), bottom_shelf_array-T_m2b)
    bottom_shelf.pose.position.x = bottom_shelf_array[0]
    bottom_shelf.pose.position.y = bottom_shelf_array[1] 
    bottom_shelf.pose.position.z = bottom_shelf_array[2] + 0.09

    top_shelf_array = np.array([top_shelf.pose.position.x, top_shelf.pose.position.y, top_shelf.pose.position.z])
    top_shelf_array = np.dot(np.transpose(R_m2b), top_shelf_array-T_m2b)
    top_shelf.pose.position.x = top_shelf_array[0]
    top_shelf.pose.position.y = top_shelf_array[1] - 0.09
    top_shelf.pose.position.z = top_shelf_array[2] 

    # bottom_shelf_array = np.array([bottom_shelf.pose.position.x, bottom_shelf.pose.position.y, bottom_shelf.pose.position.z])
    # bottom_shelf_array = np.dot(np.transpose(R_m2b), bottom_shelf_array-T_m2b)
    # bottom_shelf.pose.position.x = bottom_shelf_array[0]
    # bottom_shelf.pose.position.y = bottom_shelf_array[1]
    # bottom_shelf.pose.position.z = bottom_shelf_array[2]

    moveit_commander.roscpp_initialize(sys.argv) 
    scene = moveit_commander.PlanningSceneInterface()
    robot = moveit_commander.RobotCommander()
    arm_torso_group = moveit_commander.MoveGroupCommander("arm_torso")
    arm_group = moveit_commander.MoveGroupCommander("arm")
    gripper = moveit_commander.MoveGroupCommander("gripper")

    # trophy.pose.position.y -=0.25
    # trophy.pose.position.x -= 0.2
    #return_init(arm_group, gripper)
    grab(arm_group, gripper, trophy)
    
    drop(arm_torso_group, gripper, middle_shelf)

    grab(arm_group, gripper, cup)

    drop(arm_torso_group, gripper, bottom_shelf)

    #return_init(arm_group, gripper)



if __name__ == '__main__':
    rospy.init_node('listener', anonymous=True)
    listener()

