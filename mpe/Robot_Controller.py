#!/usr/bin/env python3
import rospy
import math
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

rospy.init_node(f"robot_control_node2")
rospy.loginfo(f"robot control node started")


class Velocity():
    def __init__(self,x=0,y=0,z=0):
        self.x = x
        self.y = y
        self.z = z
    def __str__(self):
        out = f"(x:{self.x},y:{self.y},z:{self.z})"
        return out

class Robot_Controller():
    def __init__(self,robotid=1, init_pos = [0,0]):
        # rospy.init_node(f"robot_control_node{robotid}")
        # rospy.loginfo(f"robot control node started")
        self.id = robotid
        self.position=init_pos
        self.orientation=0
        self.linear = Velocity()
        self.angular = Velocity()
        
        # rospy.init_node(f"robot_control_node{robotid}")
        # rospy.loginfo(f"robot control node {robotid} started")
        
        
        self.pub = rospy.Publisher(f"Robot{robotid}/cmd_vel",Twist,queue_size=10)
        sub = rospy.Subscriber(f"Robot{robotid}/odom", Odometry, callback=self.odom_callback)

    def odom_callback(self,msg : Odometry):
        position = msg.pose.pose.position
        self.position=[position.y,position.x]            #Flipped to align with petting zoo axis
        q = msg.pose.pose.orientation
        siny_cosp = 2*(q.w*q.z+q.x*q.y)
        cosy_cosp = 1-2*(q.y*q.y+q.z*q.z)
        self.orientation = math.atan2(siny_cosp,cosy_cosp)
        # rospy.loginfo(f"orientation: {self.orientation}")
        # rospy.loginfo(f"Position: {self.position}")
        # rospy.loginfo(f"Linear Velocity of Robot{self.id}:{self.linear}")
        # rospy.loginfo(f"In robot{self.id}: publisher {self.pub.name}")
        cmd = Twist()
        cmd.linear.x = self.linear.x
        cmd.linear.y = self.linear.y
        cmd.linear.z = self.linear.z
        cmd.angular.x = self.angular.x
        cmd.angular.y = self.angular.y
        cmd.angular.z = self.angular.z
        self.pub.publish(cmd)

    def update_velocity_petting_zoo(self, action, sim_heading):
        # Takes in petting zoo action, [x y] velocities
        # and simulation heading of robot
        # Note that upper left corner of 2-d sim is 0,0, so positive velocities and headings are to the right and downwards
        # cmd = Twist()
        sim_heading = math.atan2(sim_heading[0],sim_heading[1])
        # rospy.loginfo(f"Simulation Action for Robot{self.id}: {action} \n \
        #               Sim Heading for Robot{self.id}: {sim_heading} \n \
        #               Gazebo Heading for Robot{self.id}: {self.orientation}")
        rate=0.7
        self.linear.x = 0.1
        if abs(sim_heading-self.orientation)<0.5:
            self.angular.z=0
        elif sim_heading>self.orientation and sim_heading<(self.orientation+math.pi):
            self.angular.z=rate
        elif sim_heading<self.orientation and sim_heading>(self.orientation-math.pi):
            self.angular.z=-rate
        elif sim_heading>self.orientation:
            self.angular.z=-rate
        else:
            self.angular.z=rate
        # self.pub.publish(cmd)

    def halt(self):
        self.linear = Velocity()
        self.angular = Velocity()

    def update_velocity(self,linear = Velocity(),angular=Velocity()):
        self.linear = linear
        self.angular = angular

    def get_orientation(self):
        return self.orientation
    
    def get_position(self):
        return self.position

if __name__ == "__main__":
    rospy.init_node(f"robot_control_node")
    rospy.loginfo(f"robot control node started")

    robot1 = Robot_Controller(1)
    robot2 = Robot_Controller(2)
    robot3 = Robot_Controller(3)
    robot4 = Robot_Controller(4)
    robot5 = Robot_Controller(5)
    rospy.sleep(2)
    robot1.update_velocity(Velocity(0,0,0))
    robot2.update_velocity(Velocity(),Velocity(0,0,0.2))
    robot3.update_velocity(Velocity(),Velocity(0,0,-0.2))
    rospy.sleep(2)
    robot1.update_velocity(Velocity(),Velocity())
    robot2.update_velocity(Velocity(),Velocity())
    robot3.update_velocity(Velocity(),Velocity())
    robot4.update_velocity(Velocity(),Velocity())
    robot5.update_velocity(Velocity(),Velocity())
    rospy.sleep(1)