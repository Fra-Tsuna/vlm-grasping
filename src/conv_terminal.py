#!/usr/bin/env python
import os
import sys
import rospy
import speech_recognition as sr
from actionlib import SimpleActionClient
from pal_interaction_msgs.msg import TtsAction, TtsGoal
from std_msgs.msg import String

# same as conv.py, but then with commands from the terminal instead of speech commands

pub = rospy.Publisher("/text_command", String, queue_size=100)

def listener():
    rospy.init_node("listener", anonymous = True)
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        print("Ready for the next command")
        answer = raw_input()
        pub.publish(answer)
        rate.sleep()

    return

if __name__ == '__main__':
    listener()
