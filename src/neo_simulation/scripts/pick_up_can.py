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
# Gripper
group_name2 = "gripper"
move_group2 = moveit_commander.MoveGroupCommander(group_name2)
# Reference Frame
#planning_frame = move_group.get_planning_frame()
# print "======= Planning Frame: %s" % planning_frame
# End Effector
#eef_link = move_group.get_end_effector_link()
# print "======= End effector link: %s" % eef_link
# All the groups
#group_names = robot.get_group_names()
# print "======= Available Planning Groups: %s" % group_names
# State of the robot
# print "======= Printing robot state"
# print robot.get_current_state()
# print ""

# Get near
joint_goal = move_group.get_current_joint_values()

joint_goal[0] = -2.9451785280744085
joint_goal[1] = 1.0138321626356213
joint_goal[2] = 0.4091627931356854
joint_goal[3] = -1.3605316462629666
joint_goal[4] = 1.5494746828218924
joint_goal[5] = 0.19274827010193096
# Execution
move_group.go(joint_goal, wait=True)
move_group.stop()

# Open gripper
joint_goal = move_group2.set_named_target("open2")
# Execution
move_group2.go(joint_goal, wait=True)
move_group2.stop()

# Lower gripper (Beer)
joint_goal = move_group.get_current_joint_values()

joint_goal[0] = -2.9490742625133795
joint_goal[1] = 0.9330952121221108
joint_goal[2] = 0.240494168768298
joint_goal[3] = -1.1110589763458556
joint_goal[4] = 1.5495105704784935
joint_goal[5] = 0.188793683609215
# Execution
move_group.go(joint_goal, wait=True)
move_group.stop()

# Lower gripper (Coke)
joint_goal = move_group.get_current_joint_values()

joint_goal[0] = -2.9509881259836783
joint_goal[1] = 0.8687045297168909
joint_goal[2] = 0.17766993231805106
joint_goal[3] = -0.9846909553002074
joint_goal[4] = 1.5501262258370039
joint_goal[5] = 0.18680821562187688
# Execution
move_group.go(joint_goal, wait=True)
move_group.stop()
"""
# Grab
joint_goal = move_group2.set_named_target("closed3")
# Execution
move_group2.go(joint_goal, wait=True)
move_group2.stop()

# Get near
joint_goal = move_group.get_current_joint_values()

joint_goal[0] = -2.9451785280744085
joint_goal[1] = 1.0138321626356213
joint_goal[2] = 0.4091627931356854
joint_goal[3] = -1.3605316462629666
joint_goal[4] = 1.5494746828218924
joint_goal[5] = 0.19274827010193096
# Execution
move_group.go(joint_goal, wait=True)
move_group.stop()

# Return to walking form
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

"""
