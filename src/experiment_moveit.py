#!/usr/bin/env python
import rospy
import copy
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
    group_name = "arm"
    move_group = moveit_commander.MoveGroupCommander("arm")

    planning_frame = move_group.get_planning_frame()
    print "============ Planning frame: %s" % planning_frame

    # We can also print the name of the end-effector link for this group:
    eef_link = move_group.get_end_effector_link()
    print "============ End effector link: %s" % eef_link

    # We can get a list of all the groups in the robot:
    group_names = robot.get_group_names()
    print "============ Available Planning Groups:", robot.get_group_names()

    # Sometimes for debugging it is useful to print the entire state of the
    # robot:
    print "============ Printing robot state"
    print robot.get_current_state()
    print ""
  
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
    print(waypoints)
    new_wp = copy.deepcopy(waypoints)

    # We want the Cartesian path to be interpolated at a resolution of 1 cm
    # which is why we will specify 0.01 as the eef_step in Cartesian
    # translation.  We will disable the jump threshold by setting it to 0.0,
    # ignoring the check for infeasible jumps in joint space, which is sufficient
    # for this tutorial.
    (plan, fraction) = move_group.compute_cartesian_path(
                                    waypoints,   # waypoints to follow
                                    0.01,        # eef_step
                                    0.0)    

    move_group.execute(plan, wait=True)
    move_group.stop()
    move_group.clear_pose_targets()
    print("qua")
    new_wp.reverse()
    print(new_wp)

    # nuovi_wp = waypoints.reverse()
    (plan, fraction) = move_group.compute_cartesian_path(
                                    new_wp,   # waypoints to follow
                                    0.01,        # eef_step
                                    0.0)
        
    move_group.execute(plan, wait=True)
    move_group.stop() 
    move_group.clear_pose_targets()                         

    # rospy.spin()

if __name__ == '__main__':
    listener()
