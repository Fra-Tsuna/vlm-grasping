#!/usr/bin/env python

import rospy
import numpy as np

from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from PIL import Image as PILImage

import cPickle as pickle  # For Python 2.x
from open3d_ros_helper import open3d_ros_helper as orh
import open3d as o3d
import cv2
import io
import base64
import os
import time
import tf2_ros
import tf2_py as tf2

rospy.init_node('image_processing', anonymous=True)
bridge = CvBridge()
tf_buffer = tf2_ros.Buffer()
tf_listener = tf2_ros.TransformListener(tf_buffer)

ROOT_DIR = os.path.abspath(__file__+'/../..')
SCAN_DIR = ROOT_DIR+'/images/test_order/'

CONFIG_DIR = ROOT_DIR+'/config/dump_order/'

def depth_image_to_point_cloud(depth_image, camera_intrinsics):

    height, width = depth_image.shape
    points = []

    v, u =  np.indices((height, width))

    x = (u - camera_intrinsics[0, 2]) * depth_image / camera_intrinsics[0, 0]
    y = (v - camera_intrinsics[1, 2]) * depth_image / camera_intrinsics[1, 1]
    z = depth_image

    points = np.dstack((x, y, z)).reshape(-1, 3)

    return points


def listener():
    msg_img = rospy.wait_for_message("/xtion/rgb/image_rect_color", Image)
    img = bridge.imgmsg_to_cv2(msg_img, "bgr8")
    img_path = SCAN_DIR+"{type}.jpg".format(**{"type": "rgb"})
    cv2.imwrite(img_path, img)
    # print("quo")

    msg_img_g = rospy.wait_for_message("/xtion/depth/image_raw", Image)
    camera_info = rospy.wait_for_message("/xtion/depth/camera_info", CameraInfo)
    proj_matrix = camera_info.K   

    fx = proj_matrix[0]
    fy = proj_matrix[4]
    cx = proj_matrix[2]
    cy = proj_matrix[5]


    camera_intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    img_g = bridge.imgmsg_to_cv2(msg_img_g)
    depth_image = np.asarray(img_g)

    point_cloud = depth_image_to_point_cloud(depth_image, camera_intrinsics)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)

    pcd.transform(np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]]))
  
    o3d.visualization.draw_geometries([pcd])
    o3d.io.write_point_cloud(CONFIG_DIR+"test_pcl.pcd", pcd)

if __name__ == '__main__':
    listener()

    