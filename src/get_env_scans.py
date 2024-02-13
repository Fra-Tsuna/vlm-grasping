#!/usr/bin/env python

import rospy
import numpy as np

from circular_buffer import CircularBuffer, View
from geometry_msgs.msg import Twist, Vector3
from sensor_msgs.msg import Image
from sprockets.clients import memcached

from subprocess import call
from cv_bridge import CvBridge
from PIL import Image as PILImage

import cv2
import io
import base64
import os
import time

bridge = CvBridge()

rotation_publisher = rospy.Publisher("/mobile_base_controller/cmd_vel", Twist, queue_size=100)
# # # client_publisher = rospy.Publisher("/client_image_topic", Image, queue_size=100)

ROTATION_ANGLE = 60 #degrees
ROTATION_VELOCITY_Z = ROTATION_ANGLE*np.pi/180 #rad/s 
ROOT_DIR = os.path.abspath(__file__+'/../..')
SCAN_DIR = ROOT_DIR+'/images/scans/'

print(ROOT_DIR)
print(SCAN_DIR)

def listener():
    angle = 0
    cbuffer = CircularBuffer(int(360/ROTATION_ANGLE))

    rospy.sleep(1)
    while angle < 360:

        msg_img = rospy.wait_for_message("/xtion/rgb/image_raw", Image)
        # # # client_publisher.publish(msg_img)
        img = bridge.imgmsg_to_cv2(msg_img, "bgr8")
        img_path = SCAN_DIR+"{angle}.jpg".format(**{"angle": angle})
        print(img_path)
        cv2.imwrite(img_path, img)

        # captions = get_gpt_captions(cv2_img)
        timeout = time.time() + 1
        while True:
            rotation_publisher.publish(Twist(Vector3(0,0,0), Vector3(0,0, ROTATION_VELOCITY_Z)))
            if time.time() > timeout:
                break
        rospy.sleep(1)

        angle += ROTATION_ANGLE
    
    rospy.spin()

    # for i in range(len(cbuffer)):
    #     print("--------------")
    #     print(cbuffer[i].captions)

if __name__ == '__main__':
    rospy.init_node('get_env_scans', anonymous=True)
    
    listener()