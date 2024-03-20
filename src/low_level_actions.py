
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
GRABBING_QUATERNION = Quaternion(-0.742818371853,-0.0634568007104,0.0418472405416,0.665163821431)
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

def reach_waypoints(group,waypoints):
    (plan, fraction) = group.compute_cartesian_path(
                           waypoints,   # waypoints to follow
                           0.01,        # eef_step
                           0.0)         # jump_threshold
    group.execute(plan, wait=True)
    group.stop()
    group.clear_pose_targets()

def open_grippers(gripper):
    gripper.go([0.04, 0.04], wait=True)
    gripper.stop()
    gripper.clear_pose_targets()

def grab(group, gripper, object):
    init_pose = group.get_current_pose().pose
    print(init_pose)
    wp1 = copy.deepcopy(init_pose)
    wp1.position.z = object.pose.position.z
    wp1.position.y = object.pose.position.y 
    # wp2.position.x = object.pose.position.x 
    wp3 = copy.deepcopy(wp1)
    wp3.position.x = object.pose.position.x 

    # pose = Pose()
    # pose.position.x = object.pose.position.x - GRIPPER_OFFSET
    # pose.position.y = object.pose.position.y
    # pose.position.z = object.pose.position.z
    # pose.orientation = GRABBING_QUATERNION
    # wp3 = pose
    # print(wp3)

    # waypoints = [wp1]
    print(object.pose)
    waypoints = [wp1,wp3]
    # reversed_waypoints = [wp1, init_pose]
    reversed_waypoints = [wp1, init_pose]
    
    reach_waypoints(group, waypoints)
    close_grippers(gripper)
    reach_waypoints(group, reversed_waypoints)
    

def drop(group, gripper, goal_pose):
    init_pose = group.get_current_pose().pose
    
    wp1 = copy.deepcopy(init_pose)
    wp1.position.z = goal_pose.pose.position.z
    wp1.position.y = goal_pose.pose.position.y 

    wp3 = copy.deepcopy(wp1)
    wp3.position.x = goal_pose.pose.position.x 
    #wp2.position.x = goal_pose.pose.position.x 


    # pose = Pose()
    # pose.position.x = goal_pose.pose.position.x - GRIPPER_OFFSET
    # pose.position.y = goal_pose.pose.position.y
    # pose.position.z = goal_pose.pose.position.z
    # pose.orientation = GRABBING_QUATERNION
    # wp3 = pose

    waypoints = [wp1,wp3]
    reversed_waypoints = [ wp1, init_pose]
    
    reach_waypoints(group, waypoints)
    open_grippers(gripper)
    reach_waypoints(group, reversed_waypoints)
    
# def down(group,z):
#     init_pose = group.get_current_pose().pose
#     pose = init_pose
#     pose.position.z -= z
#     reach_pose(group, pose)

# def up(group,z):
#     init_pose = group.get_current_pose().pose
#     pose = init_pose
#     pose.position.z += z
#     reach_pose(group, pose)    

# def right(group, y):
#     init_pose = group.get_current_pose().pose
#     pose = init_pose
#     pose.position.y -= y
#     reach_pose(group, pose)    

# def left(group, y):
#     init_pose = group.get_current_pose().pose
#     pose = init_pose
#     pose.position.y += y
#     reach_pose(group, pose)    

# def forward(group, x):
#     init_pose = group.get_current_pose().pose
#     pose = init_pose
#     pose.position.x += x
#     pose.position.x -= GRIPPER_OFFSET
#     pose.orientation = GRABBING_QUATERNION
#     reach_pose(group, pose)    

# def back(group, x):
#     init_pose = group.get_current_pose().pose
#     pose = init_pose
#     pose.position.x -= x
#     pose.position.x += GRIPPER_OFFSET
#     pose.orientation = GRABBING_QUATERNION
#     reach_pose(group, pose)    