#!/usr/bin/env python
import rospy
import sys
import geometry_msgs.msg
import moveit_commander
import moveit_msgs.msg
from math import pi

display_trajectory_publisher = rospy.Publisher(
    "/move_group/display_planned_path",
    moveit_msgs.msg.DisplayTrajectory,
    queue_size=20,
)


def listener():
    moveit_commander.roscpp_initialize(sys.argv) 
    rospy.init_node('listener', anonymous=True)

    robot = moveit_commander.RobotCommander()
    scene = moveit_commander.PlanningSceneInterface()
    group_name = "arm_torso"
    print(group_name)
    move_group = moveit_commander.MoveGroupCommander("arm_torso")

    waypoints = []
    pose2 = geometry_msgs.msg.Pose()
    pose2.orientation.w = 1.0
    pose2.position.x = 0.6
    pose2.position.y = -0.3
    pose2.position.z = 1.2
    waypoints.append(pose2)

    pose1 = geometry_msgs.msg.Pose()
    pose1.orientation.w = 1.0
    pose1.position.x = 0.4
    pose1.position.y = -0.4
    pose1.position.z = 0.8
    waypoints.append(pose1)
            
    (plan, fraction) = move_group.compute_cartesian_path(
                                  waypoints,   # waypoints to follow
                                  0.01,        # eef_step
                                  0.0)         # jump_threshold

    # Move arm to the initial position
    init_pos = [0, 0, 0, 0, 0, 0, 0, 0]
    init_pos[0] = 0.14981065303305305
    init_pos[1] = 0.20002269675358963
    init_pos[2] = -1.3388411804759457
    init_pos[3] = -0.19989991663646833
    init_pos[4] = 1.937794408266278
    init_pos[5] = -1.5704218807524999
    init_pos[6] = 1.370305680517471
    init_pos[7] = 0.0002550003350565433

    move_group.execute(plan, wait=True)
    move_group.stop()
    move_group.clear_pose_targets()
    move_group.go(init_pos, wait=True)
    move_group.stop() 
    move_group.clear_pose_targets()                         

    #rospy.spin()

if __name__ == '__main__':
    listener()
