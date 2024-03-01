#!/usr/bin/env python

import rospy
from sensor_msgs.msg import PointCloud2, PointField
import std_msgs.msg
import open3d as o3d
from open3d_ros_helper import open3d_ros_helper as orh
import numpy as np
import os
import tf2_ros
import tf2_py as tf2
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
from conversion_utils import *

rospy.init_node('open3d_to_ros_publisher')


ROOT_DIR = os.path.abspath(__file__+'/../..')
SCAN_DIR = ROOT_DIR+'/images/test_order/'

CONFIG_DIR = ROOT_DIR+'/config/dump_order/'

def main():
    pub = rospy.Publisher('pointcloud_topic', PointCloud2, queue_size=10)
    rate = rospy.Rate(10)  # 10Hz

    # Load your Open3D point cloud
    pointcloud = o3d.io.read_point_cloud(CONFIG_DIR+"colored_pcl.pcd")

    for idx, p in enumerate(pointcloud.points):
        pointcloud.points[idx] = pointcloud.points[idx] / 1000
    pointcloud.transform(np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]]))

    while not rospy.is_shutdown():
        # Convert Open3D point cloud to ROS PointCloud2 message

        pcd = convertCloudFromOpen3dToRos(pointcloud, frame_id='xtion_rgb_optical_frame')
        pub.publish(pcd)
        rate.sleep()

if __name__ == '__main__':
    main()