#!/usr/bin/env python

import rospy
from std_msgs.msg import Bool
from dbw_mkz_msgs.msg import ThrottleCmd, SteeringCmd, BrakeCmd, SteeringReport
from geometry_msgs.msg import TwistStamped
import math

from twist_controller import Controller as TwistController
from yaw_controller import YawController

'''
You can build this node only after you have built (or partially built) the `waypoint_updater` node.

You will subscribe to `/twist_cmd` message which provides the proposed linear and angular velocities.
You can subscribe to any other message that you find important or refer to the document for list
of messages subscribed to by the reference implementation of this node.

One thing to keep in mind while building this node and the `twist_controller` class is the status
of `dbw_enabled`. While in the simulator, its enabled all the time, in the real car, that will
not be the case. This may cause your PID controller to accumulate error because the car could
temporarily be driven by a human instead of your controller.

We have provided two launch files with this node. Vehicle specific values (like vehicle_mass,
wheel_base) etc should not be altered in these files.

We have also provided some reference implementations for PID controller and other utility classes.
You are free to use them or build your own.

Once you have the proposed throttle, brake, and steer values, publish it on the various publishers
that we have created in the `__init__` function.

'''

MIN_SPEED = 5.
DEBUG = False

class DBWNode(object):
    def __init__(self):
        rospy.init_node('dbw_node', log_level=rospy.INFO)

        decel_limit = rospy.get_param('~decel_limit', -5)
        accel_limit = rospy.get_param('~accel_limit', 1.)
        wheel_base = rospy.get_param('~wheel_base', 2.8498)
        steer_ratio = rospy.get_param('~steer_ratio', 14.8)
        max_lat_accel = rospy.get_param('~max_lat_accel', 3.)
        max_steer_angle = rospy.get_param('~max_steer_angle', 8.)

        self.steer_pub = rospy.Publisher('/vehicle/steering_cmd', SteeringCmd, queue_size=1)
        self.throttle_pub = rospy.Publisher('/vehicle/throttle_cmd', ThrottleCmd, queue_size=1)
        self.brake_pub = rospy.Publisher('/vehicle/brake_cmd', BrakeCmd, queue_size=1)

        self.controller = TwistController(
            YawController(
                wheel_base,
                steer_ratio,
                MIN_SPEED,
                max_lat_accel,
                max_steer_angle),
            accel_limit,
            decel_limit)

        rospy.Subscriber('/vehicle/dbw_enabled', Bool, self.enabled_cb)
        rospy.Subscriber('/twist_cmd', TwistStamped, self.twist_cb)
        rospy.Subscriber('/current_velocity', TwistStamped, self.velocity_cb)
        self.dbw_enabled = False
        self.velocity = None
        self.proposed_linear = None
        self.proposed_angular = None

        self.loop()

    def enabled_cb(self, msg):
        self.dbw_enabled = msg.data
        if DEBUG:
            rospy.loginfo("DBW enabled = " + str(self.dbw_enabled))

    def twist_cb(self, msg):
        self.proposed_linear = msg.twist.linear.x
        self.proposed_angular = msg.twist.angular.z
        if DEBUG:
            rospy.logdebug("Twist update: angular = %d; linear = %d" % (self.proposed_angular, self.proposed_linear))

    def velocity_cb(self, msg):
        self.velocity = msg.twist.linear.x
        if DEBUG:
            rospy.logdebug("Velocity update: %d" % self.velocity)

    def loop(self):
        rate = rospy.Rate(50)  # 50Hz
        while not rospy.is_shutdown():
            if self.velocity is not None \
                    and self.proposed_linear is not None \
                    and self.proposed_angular is not None:

                throttle, brake, steer = self.controller.control(
                    self.proposed_linear,
                    self.proposed_angular,
                    self.velocity,
                    self.dbw_enabled)

                if self.dbw_enabled:
                    self.publish(throttle, brake, steer)

            rate.sleep()

    def publish(self, throttle, brake, steer):
        if DEBUG:
            rospy.loginfo("Controls: %.15f:%.15f:%.15f" % (throttle, brake, steer))

        tcmd = ThrottleCmd()
        tcmd.enable = True
        tcmd.pedal_cmd_type = ThrottleCmd.CMD_PERCENT
        tcmd.pedal_cmd = throttle
        self.throttle_pub.publish(tcmd)

        scmd = SteeringCmd()
        scmd.enable = True
        scmd.steering_wheel_angle_cmd = steer
        self.steer_pub.publish(scmd)

        bcmd = BrakeCmd()
        bcmd.enable = True
        bcmd.pedal_cmd_type = BrakeCmd.CMD_TORQUE
        bcmd.pedal_cmd = brake
        self.brake_pub.publish(bcmd)


if __name__ == '__main__':
    DBWNode()
