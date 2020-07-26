#!/usr/bin/env python

import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from math import pi
from moveit_commander.conversions import pose_to_list
from std_msgs.msg import String
import tf

moveit_commander.roscpp_initialize(sys.argv)
rospy.init_node('move_arm', anonymous=True)

robot = moveit_commander.RobotCommander()
scene = moveit_commander.PlanningSceneInterface()

# Arm
group_name = "arm"
move_group = moveit_commander.MoveGroupCommander(group_name)
display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
                                               moveit_msgs.msg.DisplayTrajectory,
                                               queue_size=20)

move_group.set_named_target("initial")
move_group.go(wait=True)
move_group.stop()

# Normal Walking
joint_goal = move_group.get_current_joint_values()

joint_goal[0] = -1.399701598925633
joint_goal[1] = 1.9379987099449103
joint_goal[2] = 0.45190933901079156
joint_goal[3] = -2.3630362305798487
joint_goal[4] = 1.5049051365879462
joint_goal[5] = 1.7179263804730773
# Execution
move_group.go(joint_goal, wait=True)
move_group.stop()