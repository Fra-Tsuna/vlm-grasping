#!/usr/bin/env python

import rospy
import open3d as o3d
import os
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
import numpy as np
import pickle
from matplotlib.colors import to_rgb
import tf2_ros
from conversion_utils import convertCloudFromOpen3dToRos

ROOT_DIR = os.path.abspath(__file__+'/../..')
SCAN_DIR = ROOT_DIR+'/images/test_order/'

CONFIG_DIR = ROOT_DIR+'/config/dump_order/'

COLORS = ['red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 'white', 'black']

publisher = rospy.Publisher("/pcl_centroids", MarkerArray, queue_size=100)
def listener():
    
    pcd = o3d.io.read_point_cloud(CONFIG_DIR+'colored_pcl.pcd')
    
    pcd_converted = convertCloudFromOpen3dToRos(pcd, 'xtion_rgb_optical_frame')

    with open(CONFIG_DIR+'dict/colors_dict.pkl', 'rb') as f:
        color_dict = pickle.load(f)

    array = MarkerArray()

    transform = np.array([[1,0,0],[0,-1,0],[0,0,-1]])

    id = 0
    for color, list_points in color_dict.items():
        marker = Marker()
        centroid = np.mean(list_points, axis=0)/1000
        centroid = np.dot(transform, centroid)

        marker.header.frame_id = "xtion_rgb_optical_frame"
        marker.header.stamp = rospy.Time(0)
        marker.pose.position.x = centroid[0]
        marker.pose.position.y = centroid[1]
        marker.pose.position.z = centroid[2]
        marker.pose.orientation.x = 0
        marker.pose.orientation.y = 0
        marker.pose.orientation.z = 0
        marker.pose.orientation.w = 1
        marker.type = marker.SPHERE
        color = to_rgb(color)
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = 1.0
        marker.scale.x = 0.05
        marker.scale.y = 0.05
        marker.scale.z = 0.05
        marker.id = id
        id += 1
        
        array.markers.append(marker)

    while not rospy.is_shutdown():
        publisher.publish(array)
        rate.sleep()



if __name__ == '__main__':

    rospy.init_node('spawn_clusters_centroid', anonymous=True)
    rate=rospy.Rate(10)

    
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)

    listener()