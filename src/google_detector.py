#!/usr/bin/env python

import rospy
from std_msgs.msg import String
from vlm_grasping.msg import Entity
import speech_recognition as sr

from subprocess import call

import os
import io
import numpy as np
import base64

from PIL import Image
from cv_bridge import CvBridge

from sensor_msgs.msg import Image as ImageMsg
from sprockets.clients import memcached


last_img = None
bridge = CvBridge()
pub_location = rospy.Publisher("/detected_and_filtered_entities", Entity, queue_size=100)


def convert(msg):
    cv2_img = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
    pil_im = Image.fromarray(cv2_img)
    output = io.BytesIO()
    pil_im.save(output, format="png")

    image_as_string = base64.b64encode(output.getvalue())
    return image_as_string

def update_img(msg):
    global last_img
    last_img = convert(msg)

def invoke_core():
    client.set('image_n', last_img)

    exit_code = call("/usr/bin/python3 /home/user/ws/src/vlm_grasping/src/google_detector_core.py", shell=True)
    objs_list = client.get('objs_with_bb')

    return objs_list


def listener():
    print("node started")
    rospy.Subscriber('/xtion/rgb/image_raw', ImageMsg, update_img)
    while not rospy.is_shutdown():
        frase = "h"
        frase = raw_input('Keyword: ')
        if str(frase) == "h":
            objs_list = invoke_core()
        else: 
             print("Invalid Keyword, retry.")

        for i, object in enumerate(objs_list):
            msg = Entity()
            msg.object_name = object["label"]
            coords = object["bbox"]
            msg.x_min = coords[0]
            msg.y_min = coords[1]
            msg.x_max = coords[2]
            msg.y_max = coords[3]
            
            pub_location.publish(msg)


if __name__ == '__main__':
    rospy.init_node('google_detector', anonymous=True)

    os.environ['CD_MEMCACHED_SERVERS'] = '127.0.0.1:11211<:1000M>'
    client = memcached.Client('cloud_detector')

    listener()
