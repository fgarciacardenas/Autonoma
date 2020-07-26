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
"""
# Get near left side
joint_goal = move_group.get_current_joint_values()

joint_goal[0] = -2.558130252071188
joint_goal[1] = 0.7350009934139106
joint_goal[2] = -0.03804311839000185
joint_goal[3] = -0.5915416169342533
joint_goal[4] = 1.5158203891104876
joint_goal[5] = 0.5571449306141849
# Execution
move_group.go(joint_goal, wait=True)
move_group.stop()

# Get near right side
joint_goal = move_group.get_current_joint_values()

joint_goal[0] = -3.129344533247571
joint_goal[1] = 0.8087802451319774
joint_goal[2] = -0.15098641930496282
joint_goal[3] = -0.54030254511623
joint_goal[4] = 1.5290339258983519
joint_goal[5] = 0.3095855835621748
# Execution
move_group.go(joint_goal, wait=True)
move_group.stop()

# Open Gripper
joint_goal = move_group2.set_named_target("open2")
# Execution
move_group2.go(joint_goal, wait=True)
move_group2.stop()
"""
joint_goal = move_group.get_current_joint_values()
print(joint_goal)