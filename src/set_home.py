#!/usr/bin/env python
import rospy
import sys
import geometry_msgs.msg
import moveit_commander
import moveit_msgs.msg
from math import pi

# display_trajectory_publisher = rospy.Publisher(
#     "/move_group/display_planned_path",
#     moveit_msgs.msg.DisplayTrajectory,
#     queue_size = 100,
# )


def listener():
    moveit_commander.roscpp_initialize(sys.argv) 
    rospy.init_node('listener', anonymous=True)

    robot = moveit_commander.RobotCommander()
    scene = moveit_commander.PlanningSceneInterface()
    group_name = "arm_torso"
    move_group = moveit_commander.MoveGroupCommander(group_name)

    # Move arm to the initial position
    init_pos = [0, 0, 0, 0, 0, 0, 0, 0]
    init_pos[0] = 0.35 #torso height
    init_pos[1:] = [0.2, -1.34, -0.2, 1.94, -1.57, 1.37, 0.0] # arms

    move_group.go(init_pos, wait=True)
    move_group.stop() 
    move_group.clear_pose_targets()                         

    gripper = moveit_commander.MoveGroupCommander("gripper")
    gripper.go([0.001, 0.002], wait=True)
    gripper.stop()
    gripper.clear_pose_targets()

    #rospy.spin()

if __name__ == '__main__':
    listener()
