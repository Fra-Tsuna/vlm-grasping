#!/usr/bin/env python
# coding=utf-8
import rospy
from sensor_msgs.msg import PointCloud2
from darknet_ros_msgs.msg import BoundingBoxes
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import tf2_ros
import tf2_py as tf2
import cv2
import copy
from geometry_msgs.msg import Point
import random
from nav_msgs.msg import Odometry
from control_msgs.msg import JointTrajectoryControllerState
from actionlib import SimpleActionClient
from pal_interaction_msgs.msg import TtsAction, TtsGoal

pub_list = rospy.Publisher("/detected_objects2", MarkerArray, queue_size=100)
pub_poi = rospy.Publisher("/detected_objects_poi", Marker, queue_size=100)
pub_list2 = rospy.Publisher("/object_labels", MarkerArray, queue_size=100)
pub_last_obj = rospy.Publisher("/last_detected_object", Marker, queue_size=100)
detected_list = MarkerArray()
new_marker_list = MarkerArray()

ENABLE_WINDOW = False
MIN_DETECTION_PROB = 0.01
#MIN_DETECTION_PROB = 0.5
MIN_INTERSECTION_AREA = 0.3
#MIN_INTERSECTION_AREA = 0.0001
MIN_INTERSECTION_VOL = 0.0001
MAX_DETECT_VEL = 0.05
SPEECH = rospy.get_param("/speech")                     # Turn off in simulation

rospy.init_node('listener', anonymous=True)
tf_buffer = tf2_ros.Buffer()
tf_listener = tf2_ros.TransformListener(tf_buffer)

def DetectedObjectsCallback(yolo_objects):
    odom = rospy.wait_for_message("/mobile_base_controller/odom", Odometry)
    head = rospy.wait_for_message("/head_controller/state", JointTrajectoryControllerState)
    head_joint1_vel = head.actual.velocities[0]
    head_joint2_vel = head.actual.velocities[1]
    linear_vel = odom.twist.twist.linear.x
    angular_vel = odom.twist.twist.angular.z

    # Detecting while moving is not very accurate
    if linear_vel > MAX_DETECT_VEL or angular_vel > MAX_DETECT_VEL or head_joint1_vel > MAX_DETECT_VEL or head_joint2_vel > MAX_DETECT_VEL:
        print "moving too fast to detect accurately"
        pub_list.publish(detected_list)
        pub_list2.publish(new_marker_list)
        return
 
    # Messages to wait for
    depth_objects = rospy.wait_for_message("/centroid_objects_list", MarkerArray)
    depth_camera_info = rospy.wait_for_message("/xtion/depth_registered/camera_info", CameraInfo)
    img = rospy.wait_for_message("/darknet_ros/detection_image", Image)
    
    # Window to test if the transformation of pointcloud to camera works
    if ENABLE_WINDOW:
        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(img, desired_encoding='passthrough')
        window_name = 'Testing detection'    
        color_yolo = (255, 0, 0)
        color_marker = (0, 255, 0)
        thickness = 2

    # Projection matrix needed to transform 3d objects to camera image
    proj_matrix = depth_camera_info.K   
    fx = proj_matrix[0]
    fy = proj_matrix[4]
    cx = proj_matrix[2]
    cy = proj_matrix[5]
          
    # Tranformation of the global map frame and the camera frame
    try:
        trans = tf_buffer.lookup_transform("map", "xtion_rgb_optical_frame",  rospy.Time(0), rospy.Duration(1.0))
    except:
        return

    # Quaternion coordinates
    qx = trans.transform.rotation.x
    qy = trans.transform.rotation.y
    qz = trans.transform.rotation.z
    qw = trans.transform.rotation.w
    
    # Rotation matrix
    #R = np.array([[0,0,1],[-1,0,0],[0,-1,0]])  # used for the marrtino
    R = 2*np.array([[pow(qw,2) + pow(qx,2) - 0.5, qx*qy-qw*qz, qw*qy+qx*qz],[qw*qz+qx*qy, pow(qw,2) + pow(qy,2) - 0.5, qy*qz-qw*qx],[qx*qz-qw*qy, qw*qx+qy*qz, pow(qw,2) + pow(qz,2) - 0.5]])
    R_trans = R.transpose()

    
    Tx = trans.transform.translation.x
    Ty = trans.transform.translation.y
    Tz = trans.transform.translation.z
    T = np.array([Tx, Ty, Tz]).transpose()
 
    print "Length yolo objects: \n", len(yolo_objects.bounding_boxes)
    print "Length depth objects: \n", len(depth_objects.markers)
    
    intersect_list = np.zeros((len(yolo_objects.bounding_boxes),len(depth_objects.markers)))
    n = 0

    # compare the detected objects using yolo with the transformed pointclouds  
    for obj in yolo_objects.bounding_boxes:
        xA1 = obj.xmin
        yA1 = obj.ymin
        xA2 = obj.xmax
        yA2 = obj.ymax
        SA = (xA2-xA1)*(yA2-yA1)
        
        if ENABLE_WINDOW:
            start_point = (xA1, yA1)
            end_point = (xA2, yA2)
            cv_image = cv2.rectangle(cv_image, start_point, end_point, color_yolo, thickness)

        for marker in depth_objects.markers:
            XB1 = marker.pose.position.x + marker.points[0].x
            XB2 = marker.pose.position.x - marker.points[0].x
            YB1 = marker.pose.position.y + marker.points[0].y
            YB2 = marker.pose.position.y - marker.points[0].y
            ZB1 = marker.pose.position.z + marker.points[0].z
            ZB2 = marker.pose.position.z - marker.points[0].z

            B1 = np.array([XB1, YB1, ZB1]) - T
            B2 = np.array([XB2, YB2, ZB2]) - T
            B1_rot = np.dot(R_trans,B1)
            B2_rot = np.dot(R_trans,B2)

            XB1_rot = B1_rot[0]
            YB1_rot = B1_rot[1]
            ZB1_rot = B1_rot[2]
            XB2_rot = B2_rot[0]
            YB2_rot = B2_rot[1]
            ZB2_rot = B2_rot[2]

            xB1_1 = (fx*XB1_rot + cx*ZB1_rot)/ZB1_rot
            xB2_1 = (fx*XB2_rot + cx*ZB2_rot)/ZB2_rot
            yB1_1 = (fy*YB1_rot + cy*ZB1_rot)/ZB1_rot
            yB2_1 = (fy*YB2_rot + cy*ZB2_rot)/ZB2_rot
            
            if xB1_1 > xB2_1:
                xB1 = xB2_1
                xB2 = xB1_1
            else:
                xB1 = xB1_1
                xB2 = xB2_1

            if yB1_1 > yB2_1:
                yB1 = yB2_1
                yB2 = yB1_1
            else:
                yB1 = yB1_1
                yB2 = yB2_1

            if ENABLE_WINDOW:
                start_point = (xB1.astype(int), yB1.astype(int))
                end_point = (xB2.astype(int), yB2.astype(int))
                cv_image = cv2.rectangle(cv_image, start_point, end_point, color_marker, thickness)       

            SB = (xB2-xB1)*(yB2-yB1)
            
            SI = np.maximum(0, np.minimum(xA2,xB2)-np.maximum(xA1,xB1)) * np.maximum(0,np.minimum(yA2,yB2)-np.maximum(yA1,yB1))
            SU = SA+SB-2*SI
            intersect_list[n][marker.id] = SI/SU
        n+=1

    n = 0

    # Publish correctly detected objects
    for obj in yolo_objects.bounding_boxes:
        print "Yolo probability: " , obj.probability

        if obj.probability > MIN_DETECTION_PROB:  # Check if probability of detection is high enough 
            #print "max intersect: " , np.max(intersect_list[n][:])
            print "La probabilità di yolo è maggiore della soglia"
            print "procedo con il primo IF"
            if len(intersect_list[n][:])>0 and np.max(intersect_list[n][:]) > MIN_INTERSECTION_AREA:  # Compare yolo detection with 3d cluster projection
                print "Sono dentro il primo IF"
                obj_id = np.argmax(intersect_list[n][:])
                is_detected = False
                is_better = False
                print "Procedo con il secondo IF"
                if len(detected_list.markers)>0:
                    print "Sono dentro il secondo IF, dovrbbe essere l'ultimo "               								
                    IntersectVolumes = np.zeros(len(detected_list.markers))
                    
                    Qx1 = depth_objects.markers[obj_id].pose.position.x - depth_objects.markers[obj_id].points[0].x
                    Qx2 = depth_objects.markers[obj_id].pose.position.x + depth_objects.markers[obj_id].points[0].x
 
                    Qy1 = depth_objects.markers[obj_id].pose.position.y - depth_objects.markers[obj_id].points[0].y
                    Qy2 = depth_objects.markers[obj_id].pose.position.y + depth_objects.markers[obj_id].points[0].y

                    Qz1 = depth_objects.markers[obj_id].pose.position.z - depth_objects.markers[obj_id].points[0].z
                    Qz2 = depth_objects.markers[obj_id].pose.position.z + depth_objects.markers[obj_id].points[0].z

                    m = 0
                    for detected_marker in detected_list.markers:
                        Wx1 = detected_marker.pose.position.x - detected_marker.points[0].x
                        Wx2 = detected_marker.pose.position.x + detected_marker.points[0].x
 
                        Wy1 = detected_marker.pose.position.y - detected_marker.points[0].y
                        Wy2 = detected_marker.pose.position.y + detected_marker.points[0].y

                        Wz1 = detected_marker.pose.position.z - detected_marker.points[0].z
                        Wz2 = detected_marker.pose.position.z + detected_marker.points[0].z 
                        
                        Vol_Q = (Qx2-Qx1)*(Qy2-Qy1)*(Qz2-Qz1)
                        Vol_W = (Wx2-Wx1)*(Wy2-Wy1)*(Wz2-Wz1)
                        Vol_I = np.maximum(0, np.minimum(Qx2,Wx2)-np.maximum(Qx1,Wx1)) * np.maximum(0,np.minimum(Qy2,Wy2)-np.maximum(Qy1,Wy1)) * np.maximum(0,np.minimum(Qz2,Wz2)-np.maximum(Qz1,Wz1))
                        Vol_U = Vol_Q+Vol_W-2*Vol_I
                        IntersectVolumes[m] = Vol_I/Vol_U
                        
                        m+=1
                    
                    # Check if the object has already been detected to prevent duplicates
                    print "Intersect volumes: \n" , IntersectVolumes
                    if np.max(IntersectVolumes) > MIN_INTERSECTION_VOL:
                        is_detected = True

                        # In case it already has been detected, check if the detection is an improvement
                        if np.max(intersect_list[n][:]) > detected_list.markers[np.argmax(IntersectVolumes)].points[1].x:
                            is_better = True                     

                    print "is detected: " , is_detected
                    print "is_better: ", is_better

                    if is_detected and is_better:
                        old_id = detected_list.markers[np.argmax(IntersectVolumes)].id
                        detected_list.markers.remove(detected_list.markers[np.argmax(IntersectVolumes)])
                        new_marker_list.markers.remove(new_marker_list.markers[np.argmax(IntersectVolumes)])
                    elif is_detected and not is_better:
                        continue
                
                # saying which object has been detected
                print "object label: ", obj.Class
                if SPEECH:
                    client = SimpleActionClient('/tts', TtsAction)
                    client.wait_for_server()
                    if is_better:
                        text = "I have improved the accuracy of the " + obj.Class
                    else:      
                        text = "I have detected a " + obj.Class      
                    goal = TtsGoal()
                    goal.rawtext.text = text
                    goal.rawtext.lang_id = "en_GB"
                    # Send the goal and wait
                    client.send_goal_and_wait(goal)
                    rospy.loginfo("I'll say: " + text)

                # publish new detected object
                print "[+] DETECTING PART"
                detected_obj = copy.deepcopy(depth_objects.markers[obj_id])
                detected_obj.color.b = 1.0
                detected_obj.color.r = 0.0
                detected_obj.text = obj.Class
                print "[+] DETECTED OBJECT: " + str(detected_obj.text)

                if is_detected and is_better:
                    detected_obj.id = old_id
                else: 
                    new_id = random.randint(0,999)     
                    detected_obj.id = new_id

                p2 = Point()
                p2.x = np.max(intersect_list[n][:])
                detected_obj.points.append(p2)
                detected_list.markers.append(detected_obj)
                
                poi = copy.deepcopy(detected_obj)
                poi.color.b = 0.0
                poi.color.g = 1.0
                print "PUBLISHING OBJECT"
                pub_poi.publish(detected_obj)
                pub_last_obj.publish(detected_obj)
                print "OBJECT PUBLISHED"

                # publish label above detected object
                new_marker = copy.deepcopy(depth_objects.markers[obj_id])
                new_marker.color.b = 1.0
                new_marker.color.r = 0.0
                new_marker.type = marker.TEXT_VIEW_FACING
                new_marker.text = obj.Class
                new_marker.scale.x = 0.2
                new_marker.scale.y = 0.2
                new_marker.scale.z = 0.2
                new_marker.pose.position.z = depth_objects.markers[obj_id].pose.position.z + depth_objects.markers[obj_id].points[0].z + 0.1
                
                if is_detected and is_better:
                    new_marker.id = old_id
                else:
                    new_marker.id = new_id
                new_marker.points.append(p2)
                new_marker_list.markers.append(new_marker)
                
        n+=1
    
    pub_list.publish(detected_list)
    pub_list2.publish(new_marker_list)

    if ENABLE_WINDOW:
        cv2.imshow(window_name, cv_image)
        cv2.waitKey(0) 
        cv2.destroyAllWindows() 
    

def listener():
    
    rospy.Subscriber("/darknet_ros/bounding_boxes", BoundingBoxes, DetectedObjectsCallback)
    
    rospy.spin()

if __name__ == '__main__':
    listener()
