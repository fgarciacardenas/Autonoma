#!/usr/bin/env python

import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from neo_simulation.msg import Command
from math import pi
from moveit_commander.conversions import pose_to_list
from std_msgs.msg import String
import tf

class CanObject(object):
    def __init__(self):
        self.can_type = ""
        self.command = 0
        topic = '/can_command'
        self.subs = rospy.Subscriber(topic, Command, self.callback)
        
    def callback(self, msg):
        self.can_type = msg.can_type
        self.command = msg.command


def move_near():
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
    return

def move_walk():
    joint_goal = move_group.get_current_joint_values()

    joint_goal[0] = -1.3996326933397212
    joint_goal[1] = 1.9396637995828918
    joint_goal[2] = 0.4491192805199047
    joint_goal[3] = -2.3519763173741977
    joint_goal[4] = 1.5705374322278924
    joint_goal[5] = 1.7175030007400203
    # Execution
    move_group.go(joint_goal, wait=True)
    move_group.stop()
    return

def move_to_box(can_type):
    joint_goal = move_group.get_current_joint_values()
    if can_type == "coke":
        joint_goal[0] = -2.558130252071188
        joint_goal[1] = 0.7350009934139106
        joint_goal[2] = -0.03804311839000185
        joint_goal[3] = -0.5915416169342533
        joint_goal[4] = 1.5158203891104876
        joint_goal[5] = 0.5571449306141849
    elif can_type == "beer":
        joint_goal[0] = -3.129344533247571
        joint_goal[1] = 0.8087802451319774
        joint_goal[2] = -0.15098641930496282
        joint_goal[3] = -0.54030254511623
        joint_goal[4] = 1.5290339258983519
        joint_goal[5] = 0.3095855835621748
    # Execution
    move_group.go(joint_goal, wait=True)
    move_group.stop()

def open_gripper():
    joint_goal = move_group2.set_named_target("open2")
    # Execution
    move_group2.go(joint_goal, wait=True)
    move_group2.stop()
    return

def lower_gripper(can_type):
    joint_goal = move_group.get_current_joint_values()
    if can_type == "coke":
        joint_goal[0] = -2.9509881259836783
        joint_goal[1] = 0.8687045297168909
        joint_goal[2] = 0.17766993231805106
        joint_goal[3] = -0.9846909553002074
        joint_goal[4] = 1.5501262258370039
        joint_goal[5] = 0.18680821562187688
    elif can_type == "beer":
        joint_goal[0] = -2.9490742625133795
        joint_goal[1] = 0.9330952121221108
        joint_goal[2] = 0.240494168768298
        joint_goal[3] = -1.1110589763458556
        joint_goal[4] = 1.5495105704784935
        joint_goal[5] = 0.188793683609215
    # Execution
    move_group.go(joint_goal, wait=True)
    move_group.stop()
    return

def grab(can_type):
    joint_goal = move_group2.get_current_joint_values()
    if can_type == "coke":
        joint_goal[0] = 0.85
        joint_goal[1] = 0.85
        joint_goal[2] = 0.85
    elif can_type == "beer":
        joint_goal = move_group2.set_named_target("closed3")
    # Execution
    move_group2.go(joint_goal, wait=True)
    move_group2.stop()
    return


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

robot_msg = CanObject()

rate = rospy.Rate(10)
while not rospy.is_shutdown():
    if robot_msg.command == 1:
        move_near()
        open_gripper()
        lower_gripper(robot_msg.can_type)
        grab(robot_msg.can_type)
        move_near()
        move_walk()
    elif robot_msg.command == 2:
        move_to_box(robot_msg.can_type)
        open_gripper()
        move_walk()


