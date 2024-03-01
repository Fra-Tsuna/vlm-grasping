#!/usr/bin/env python
import rospy
import numpy as np
import sys
import copy
import tf2_ros
import math
import tf2_py as tf2
import moveit_commander
import keyboard
import cv2
import moveit_msgs.msg
import geometry_msgs.msg
from math import pi
from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list


from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from nav_msgs.msg import Odometry
from trajectory_msgs.msg import JointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint

from actionlib import SimpleActionClient
from pal_interaction_msgs.msg import TtsAction, TtsGoal

from control_msgs.msg import FollowJointTrajectoryGoal
from control_msgs.msg import FollowJointTrajectoryAction

SPEECH = rospy.get_param("/speech")
MAX_GRAB_VEL = 0.1 
ARM_REACH = 1.5
OFFSET_GRIPPER = 0.15

import rosservice
from std_srvs.srv import Empty


print("a")
moveit_commander.roscpp_initialize(sys.argv)
rospy.init_node('listener', anonymous=True)
print 'b'
tf_buffer = tf2_ros.Buffer()
tf_listener = tf2_ros.TransformListener(tf_buffer)
robot = moveit_commander.RobotCommander()
scene = moveit_commander.PlanningSceneInterface()
print 'c'
group_name = "arm_torso"
move_group = moveit_commander.MoveGroupCommander(group_name)
print 'd'

#display_trajectory_publisher = rospy.Publisher('/move_group/monitored_planning_scene',
                                               #moveit_msgs.msg.DisplayTrajectory,
                                               #queue_size=20)


def GraspCallback(obj): 
    print("dentro la callback?")
    odom = rospy.wait_for_message("/mobile_base_controller/odom", Odometry)
    linear_vel = odom.twist.twist.linear.x
    angular_vel = odom.twist.twist.angular.z
    # The robot can only grab bottles or cups
    if obj.text != "bottle" and obj.text != "cup":
        text = "Cannot grab a " + obj.text
        if SPEECH:
            client = SimpleActionClient('/tts', TtsAction)
            client.wait_for_server()
            goal = TtsGoal()
            goal.rawtext.text = text 
            goal.rawtext.lang_id = "en_GB"
            client.send_goal_and_wait(goal)
        else:
            print text
        return
    # The robot cannot grab when moving too fast
    if linear_vel > MAX_GRAB_VEL or angular_vel > MAX_GRAB_VEL:
        text = "moving to fast to grab"
        if SPEECH:
            client = SimpleActionClient('/tts', TtsAction)
            client.wait_for_server()
            goal = TtsGoal()
            goal.rawtext.text = text 
            goal.rawtext.lang_id = "en_GB"
            client.send_goal_and_wait(goal)
        else:
            print text
        return

    # As the transfrom sometimes fails, especially in the beginning, it is desired to use such a try statement to prevent the node from failing
    try:
        trans_base = tf_buffer.lookup_transform("map", "base_footprint",  rospy.Time(0), rospy.Duration(1.0))
        trans_arm = tf_buffer.lookup_transform("map", "arm_1_link",  rospy.Time(0), rospy.Duration(1.0))
    except:
        text = "No transform found"
        if SPEECH:
            client = SimpleActionClient('/tts', TtsAction)
            client.wait_for_server()
            goal = TtsGoal()
            goal.rawtext.text = text 
            goal.rawtext.lang_id = "en_GB"
            client.send_goal_and_wait(goal)
        else:
            print text
        return
    
    # The object is given in the "map" frame, however for the grasping the object has to be in the "base" frame
    Tx_base = trans_base.transform.translation.x
    Ty_base = trans_base.transform.translation.y
    Tz_base = trans_base.transform.translation.z
    T_base = np.array([Tx_base, Ty_base, Tz_base])
    # Quaternion coordinates
    qx = trans_base.transform.rotation.x
    qy = trans_base.transform.rotation.y
    qz = trans_base.transform.rotation.z
    qw = trans_base.transform.rotation.w
    
    # Rotation matrix
    R = 2*np.array([[pow(qw,2) + pow(qx,2) - 0.5, qx*qy-qw*qz, qw*qy+qx*qz],[qw*qz+qx*qy, pow(qw,2) + pow(qy,2) - 0.5, qy*qz-qw*qx],[qx*qz-qw*qy, qw*qx+qy*qz, pow(qw,2) + pow(qz,2) - 0.5]])
    R_trans = R.transpose()


    Tx_arm = trans_arm.transform.translation.x
    Ty_arm = trans_arm.transform.translation.y
    Tz_arm = trans_arm.transform.translation.z
    T_arm = np.array([Tx_arm, Ty_arm, Tz_arm])

    px =  obj.pose.position.x
    py =  obj.pose.position.y
    pz =  obj.pose.position.z
    p_arm = np.array([px, py, pz]) - T_arm

    p_base = np.array([px, py, pz]) - T_base
    p_base_rot = np.dot(R_trans,p_base)
    distance_arm = np.linalg.norm(p_arm)

    # check if the robot is able to reach the object. It is still possible that after this if statement it can still not reach the object, but this way many redundant computations are prevented when it is obvious the robot cannot reach it.
    if distance_arm > ARM_REACH or p_base_rot[0] < 0:
        text = "Cannot reach object"
        if SPEECH:
            client = SimpleActionClient('/tts', TtsAction)
            client.wait_for_server()
            goal = TtsGoal()
            goal.rawtext.text = text 
            goal.rawtext.lang_id = "en_GB"
            client.send_goal_and_wait(goal)
        else:
            print text
        return
    
    # Several intermediate points are used to have a nice grabbing movement                         
    waypoints = []
    
    pose1 = geometry_msgs.msg.Pose()
    pose1.orientation.w = 1.0
    pose1.position.x = 0.4
    pose1.position.y = -0.4
    pose1.position.z = 0.8
    waypoints.append(pose1)

    pose2 = geometry_msgs.msg.Pose()
    pose2.orientation.w = 1.0
    pose2.position.x = 0.6
    pose2.position.y = -0.3
    pose2.position.z = 1.2
    waypoints.append(pose2)

    # Moves the gripper above the detected object and turns the gripper to the right orientation for grasping
    pose3 = geometry_msgs.msg.Pose()
    pose3.orientation.w = 0.5*math.sqrt(2)
    pose3.orientation.x = 0.5*math.sqrt(2)
    pose3.position.x = p_base_rot[0] - OFFSET_GRIPPER
    pose3.position.y = p_base_rot[1]
    pose3.position.z = p_base_rot[2] + obj.scale.z
    waypoints.append(pose3)
            


    # Tries to find a grabbing plan
    (plan, fraction) = move_group.compute_cartesian_path(
                           waypoints,   # waypoints to follow
                           0.01,        # eef_step
                           0.0)         # jump_threshold

    #display_trajectory = moveit_msgs.msg.DisplayTrajectory()
    #display_trajectory.trajectory_start = robot.get_current_state()
    #display_trajectory.trajectory.append(plan)
    #display_trajectory_publisher.publish(display_trajectory)

    #except:
    #    text = "No trajectory found"
    #    client = SimpleActionClient('/tts', TtsAction)
    #    client.wait_for_server()
    #    goal = TtsGoal()
    #    goal.rawtext.text = text 
    #    goal.rawtext.lang_id = "en_GB"
    #    client.send_goal_and_wait(goal)
    #    return

        
    #try:
    move_group.execute(plan, wait=True)
    move_group.stop()
    move_group.clear_pose_targets()
    print "Execution succeeded!!!"

    #except:
    #    text = "Execution failed"
    #    client = SimpleActionClient('/tts', TtsAction)
    #    client.wait_for_server()
    #    goal = TtsGoal()
    #    goal.rawtext.text = text 
    #    goal.rawtext.lang_id = "en_GB"
    #   client.send_goal_and_wait(goal)
    #    return

    # Open gripper
    gripper_open = JointTrajectory()
    gripper_open.joint_names = ["gripper_left_finger_joint", "gripper_right_finger_joint"]
    gripper_point = JointTrajectoryPoint()
    gripper_point.positions = [0.04, 0.04]
    gripper_point.time_from_start.secs = 1
    gripper_point.time_from_start.nsecs = 1

    #gripper_open.points[0].positions[0] = 0.04
    #gripper_open.points[0].positions[1] = 0.04
    gripper_open.points.append(gripper_point)
    #pub_gripper.publish(gripper_open)

    client = SimpleActionClient('/gripper_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
    client.wait_for_server()
    goal = FollowJointTrajectoryGoal()
    goal.trajectory = gripper_open

    client.send_goal_and_wait(goal)


    # Moves
    pose_goal = geometry_msgs.msg.Pose()
    pose_goal.orientation.w = 0.5*math.sqrt(2)
    pose_goal.orientation.x = 0.5*math.sqrt(2)
    pose_goal.position.x = p_base_rot[0] - OFFSET_GRIPPER
    pose_goal.position.y = p_base_rot[1]
    pose_goal.position.z = p_base_rot[2]


    move_group.set_pose_target(pose_goal)

    plan = move_group.go(wait=True)
    move_group.stop()
    move_group.clear_pose_targets()

    # Grasps object till a certain force is detected.
    # It at first did not close enough to really grab the bottle in the demo, thus I made it close slightly more by changing /opt/pal/ferrum/lib/pal_parallel_gripper_wrapper/gripper_grasping.py. I added -0.01 to line 115 and 116 (using gedit)
    if SPEECH:
        rospy.wait_for_service('/parallel_gripper_controller/grasp')
        gripper_grasp = rospy.ServiceProxy('/parallel_gripper_controller/grasp', Empty)
        hallo = gripper_grasp()

    # Moves grabbed object to random position for the demo
    pose_up = geometry_msgs.msg.Pose()
    pose_up.orientation.w = 1
    pose_up.position.x = p_base_rot[0] - 0.2
    pose_up.position.y = p_base_rot[1]
    pose_up.position.z = p_base_rot[2] + 0.2
 
    move_group.set_pose_target(pose_up)

    plan = move_group.go(wait=True)
    move_group.stop()
    move_group.clear_pose_targets()

    command = String()
    # Waits still you say "put it back"
    while str(command).find("put") == -1 or str(command).find("back") == -1: 
        command = rospy.wait_for_message("/text_command", String)
        if str(command).find("put") != -1 and str(command).find("back") != -1:
            break 

    # Puts the object back
    move_group.set_pose_target(pose_goal)

    plan = move_group.go(wait=True)
    move_group.stop()
    move_group.clear_pose_targets()
   
    client = SimpleActionClient('/gripper_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
    client.wait_for_server()
    goal = FollowJointTrajectoryGoal()
    goal.trajectory = gripper_open
    client.send_goal_and_wait(goal)
    move_group.set_pose_target(pose3)
    plan = move_group.go(wait=True)
    move_group.stop()
    move_group.clear_pose_targets()

    

def listener():

    rospy.Subscriber("/grasping_object", Marker, GraspCallback)
  
    
    rospy.spin()

if __name__ == '__main__':
    listener()
