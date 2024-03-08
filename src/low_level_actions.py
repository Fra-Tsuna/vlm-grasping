
import rospy
import open3d as o3d
import os
import numpy as np
import copy
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

# GRABBING_QUATERNION =  Quaternion(-0.528,0.412,0.473,0.571)
g = Quaternion(-0.742818371853,-0.0634568007104,0.0418472405416,0.665163821431)
GRABBING_QUATERNION = g
GRIPPER_OFFSET = 0.15

def reach_pose(group,pose):
    group.set_pose_target(pose)
    group.go(wait=True)
    group.stop()
    group.clear_pose_targets()

def close_grippers(gripper):
    gripper.go([0.01, 0.02], wait=True)
    gripper.stop()
    gripper.clear_pose_targets()

def open_grippers(gripper):
    gripper.go([0.04, 0.04], wait=True)
    gripper.stop()
    gripper.clear_pose_targets()

# def grab(object):
#     arm_group = moveit_commander.MoveGroupCommander("arm_torso")
#     init_pose = arm_group.get_current_pose().pose
#     print(init_pose)
#     pose = Pose()
#     # pose.position.x = object.pose.position.x
#     # pose.position.y = object.pose.position.y
#     pose.position.z = object.pose.position.z + GRIPPER_OFFSET
#     pose.orientation = GRABBING_QUATERNION
#     reach_pose(arm_group, pose)
#     close_grippers()
#     reach_pose(arm_group, init_pose)

# def drop(goal_pose):
#     arm_group = moveit_commander.MoveGroupCommander("arm_torso")
#     init_pose = arm_group.get_current_pose().pose
#     pose = Pose()
#     pose.position = goal_pose.pose.position
#     pose.position.z += GRIPPER_OFFSET
#     pose.orientation = GRABBING_QUATERNION
#     reach_pose(arm_group, pose)
#     open_grippers()
#     reach_pose(arm_group, init_pose)
    
def down(group,z):
    init_pose = group.get_current_pose().pose
    pose = init_pose
    pose.position.z -= z
    reach_pose(group, pose)

def up(group,z):
    init_pose = group.get_current_pose().pose
    pose = init_pose
    pose.position.z += z
    reach_pose(group, pose)    

def right(group, y):
    init_pose = group.get_current_pose().pose
    pose = init_pose
    pose.position.y -= y
    reach_pose(group, pose)    

def left(group, y):
    init_pose = group.get_current_pose().pose
    pose = init_pose
    pose.position.y += y
    reach_pose(group, pose)    

def forward(group, x):
    init_pose = group.get_current_pose().pose
    pose = init_pose
    pose.position.x += x
    pose.position.x -= GRIPPER_OFFSET
    pose.orientation = GRABBING_QUATERNION
    reach_pose(group, pose)    

def back(group, x):
    init_pose = group.get_current_pose().pose
    pose = init_pose
    pose.position.x -= x
    pose.position.x += GRIPPER_OFFSET
    pose.orientation = GRABBING_QUATERNION
    reach_pose(group, pose)    