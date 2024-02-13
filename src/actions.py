#!/usr/bin/env python3
import rospy
import numpy as np
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from actionlib import SimpleActionClient
from pal_interaction_msgs.msg import TtsAction, TtsGoal
from std_msgs.msg import String
from std_msgs.msg import Bool
from geometry_msgs.msg import PoseWithCovarianceStamped
from pal_navigation_msgs.msg import GoToPOIAction
from pal_navigation_msgs.msg import GoToPOIGoal
from actionlib import SimpleActionClient
import spacy
import copy

SPEECH = rospy.get_param("/speech")	# turn off in simulation
rospy.init_node('listener', anonymous=True)
nlp = spacy.load('en_core_web_sm',disable=['ner','textcat'])
pub_grasp = rospy.Publisher("/grasping_object", Marker, queue_size=100)
OBJ_DIST = 2

# Note that due to the parsing library "spacy", python3 is used in this node

def ActionCallback(command):
    # TIAGo responds with "Thank you", when saying "Good job"   
    if str(command).find("good") and str(command).find("job") != -1:
        text = "Thank you"
        if SPEECH:
            client = SimpleActionClient('/tts', TtsAction)
            client.wait_for_server()
            goal = TtsGoal()
            goal.rawtext.text = text
            goal.rawtext.lang_id = "en_GB"
            client.send_goal_and_wait(goal)
        else:
            print(text)
        return

    detected_objects = rospy.wait_for_message("/detected_objects2", MarkerArray)

    # Loop through detected objects to see if it matches a word in the command
    for obj in detected_objects.markers:
        print(obj.text)
        current_string = obj.text
        print("current object: ", current_string)
        if str(command).find(obj.text) != -1:
            doc = nlp(str(command))	# parsing of the command into tokens
            print("DOC \n\n\n\n")
            for token in doc:
                print("[+]")
                print(token)
                # If the command contains the word "grab" and if it is related to an detected object 
                if token.text == 'grab':
                    for subtoken in token.children:
                        if subtoken.text == obj.text:
                            print("publishing object")
                            pub_grasp.publish(obj)
                            return
                # If the command contains the word "move" and a detected object
                if token.text == 'move':
                    print("detected move")
                    obj_vec = np.array((obj.pose.position.x,obj.pose.position.y))
                    POIs = rospy.wait_for_message("/topology_map_poi", MarkerArray)
                    dist_obj_list = []
                    # Loop through the points of interest to find the one that corresponds to the object from the command
                    for poi in POIs.markers:
                        poi_vec = np.array((poi.pose.position.x,poi.pose.position.y))
                        dist_obj = np.linalg.norm(poi_vec-obj_vec)
                        dist_obj_list.append(dist_obj)
                    min_dist = min(dist_obj_list)
                    min_dist_index = dist_obj_list.index(min_dist)
                    # if the object is within a 2 meter radius of a point of interest, it moves to that point
                    if min_dist < OBJ_DIST:
                        print("satisfies minimal distance")
                        point = POIs.markers[min_dist_index]
                        qw = point.pose.orientation.w
                        qz = point.pose.orientation.z
                        orientation = np.arctan2(2*qw*qz, 1-2*pow(qz,2)).item()
                        poi_move = ['submap_0', 'point', point.pose.position.x, point.pose.position.y, orientation]
                        rospy.set_param("/mmap/poi/submap_0/point", poi_move)
                                 
                        client = SimpleActionClient('/poi_navigation_server/go_to_poi', GoToPOIAction)
                        client.wait_for_server()
                        goal = GoToPOIGoal()
                        goal.poi.data = 'point'
                        client.send_goal_and_wait(goal)
                        return
    print("Sorry, no results")
    return    

def listener():
    
    rospy.Subscriber("/text_command_terminal", String, ActionCallback)
    
    rospy.spin()

if __name__ == '__main__':
    listener()
