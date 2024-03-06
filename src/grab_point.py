#!/usr/bin/env python

import rospy
import open3d as o3d
import os
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
import numpy as np
import rospy
import sys
import geometry_msgs.msg
import moveit_commander
import moveit_msgs.msg
ROOT_DIR = os.path.abspath(__file__+'/../..')
SCAN_DIR = ROOT_DIR+'/images/test_order/'

CONFIG_DIR = ROOT_DIR+'/config/dump_order/'

display_trajectory_publisher = rospy.Publisher(
    "/move_group/display_planned_path",
    moveit_msgs.msg.DisplayTrajectory,
    queue_size=20,
)
publisher = rospy.Publisher("/pcl_centroids", MarkerArray, queue_size=100)

def GrabCentroidCallback(centroid_list):
    centroid = centroid_list[0]
    moveit_commander.roscpp_initialize(sys.argv) 
    rospy.init_node('listener', anonymous=True)

    robot = moveit_commander.RobotCommander()
    scene = moveit_commander.PlanningSceneInterface()
    group_name = "arm"
    move_group = moveit_commander.MoveGroupCommander(group_name)


    goal_pose = geometry_msgs.msg.Pose()
    goal_pose.orientation.w = 1.0
    goal_pose.position.x = centroid.pose.position.x
    goal_pose.position.y = centroid.pose.position.y
    goal_pose.position.z = centroid.pose.position.z

    move_group.go(goal_pose, wait=True)
    move_group.stop() 
    move_group.clear_pose_targets()        




def listener():
    rospy.Subscriber("/pcl_centroids", MarkerArray, GrabCentroidCallback)
    

if __name__ == '__main__':
    rospy.init_node('grab_point', anonymous=True)
    listener()

