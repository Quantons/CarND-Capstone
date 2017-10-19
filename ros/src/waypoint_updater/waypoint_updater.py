#!/usr/bin/env python

import rospy
import tf
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint

import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200  # Number of waypoints we will publish. You can change this number
DEBUG = True


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb, queue_size=1)

        self.base_waypoints_sub = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb, queue_size=1)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below

        self.pos_x = 0.
        self.pos_y = 0.
        self.pos_z = 0.
        self.current_orient = None
        self.yaw = 0.
        self.waypoints = None
        self.next_wpt = None

        rospy.spin()

    def pose_cb(self, msg):
        if DEBUG:
            print('position received')

        self.pos_x = msg.pose.position.x
        self.pos_y = msg.pose.position.y
        self.pos_z = msg.pose.position.z
        self.current_orient = msg.pose.orientation
        _, _, self.yaw = tf.transformations.euler_from_quaternion([self.current_orient.x, self.current_orient.y, self.current_orient.z, self.current_orient.w])
        self.yaw *= -1.

        if self.waypoints is not None:
            start_idx = self.determine_next_wpt_idx()
            final_wpts = Lane()
            final_wpts.header.stamp = rospy.Time.now()

            final_wpt_idx = start_idx + LOOKAHEAD_WPS
            if final_wpt_idx < len(self.waypoints):  # protect against wrapping around the waypoints array
                final_wpts.waypoints = self.waypoints[start_idx:final_wpt_idx]
            else:
                final_idx = (LOOKAHEAD_WPS + start_idx) % len(self.waypoints)
                final_wpts.waypoints = self.waypoints[start_idx:len(self.waypoints)] + self.waypoints[0:final_idx]
            self.final_waypoints_pub.publish(final_wpts)

            if DEBUG:
                print "Final waypoints size = %d" % len(final_wpts.waypoints)

    def waypoints_cb(self, msg):
        self.waypoints = msg.waypoints
        if DEBUG:
            print "Received waypoints size = %d" % len(self.waypoints)

        self.base_waypoints_sub.unregister()

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        pass

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)
        for i in range(wp1, wp2 + 1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist

    def bearing_wpt_to_car(self, wpt):
        shiftx = wpt.pose.pose.position.x - self.pos_x
        shifty = wpt.pose.pose.position.y - self.pos_y

        delta_x = shiftx * math.cos(self.yaw) - shifty * math.sin(self.yaw)
        delta_y = shiftx * math.sin(self.yaw) + shifty * math.cos(self.yaw)

        return math.atan2(delta_y, delta_x)

    def distance_wpt_to_car(self, wpt):
        return math.sqrt((self.pos_x - wpt.pose.pose.position.x) ** 2 +
                         (self.pos_y - wpt.pose.pose.position.y) ** 2 +
                         (self.pos_z - wpt.pose.pose.position.z) ** 2)

    def determine_next_wpt_idx(self):
        wpts = self.waypoints
        min_distance = 1e10
        index = 0

        for wpt_idx in range(len(wpts)):
            bearing = self.bearing_wpt_to_car(self.waypoints[wpt_idx])
            dist = self.distance_wpt_to_car(self.waypoints[wpt_idx])
            if dist < min_distance and bearing > 0.0:
                index = wpt_idx
                min_distance = dist

        return index


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
