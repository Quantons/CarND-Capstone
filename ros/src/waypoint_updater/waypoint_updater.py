#!/usr/bin/env python

import rospy
import tf
from geometry_msgs.msg import PoseStamped, TwistStamped, Pose, Point
from styx_msgs.msg import Lane, Waypoint, TrafficLightArray, TrafficLight

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

LOOKAHEAD_WPS = 50  # Number of waypoints we will publish. You can change this number
STOP_LINE     = 30  # Distance from traffic light to stop line
MAX_SPEED     = 11.176
DEBUG = False


def distance(pose1, pose2):
    return math.sqrt((pose1.position.x - pose2.position.x) ** 2 + (pose1.position.y - pose2.position.y) ** 2 + (pose1.position.z - pose2.position.z) ** 2)


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        self.max_dec = rospy.get_param('~decel_limit', -5) * 0.3
        self.max_acc = rospy.get_param('~accel_limit', 1.)

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb, queue_size=1)
        self.base_waypoints_sub = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb, queue_size=1)
        rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_lights_cb)
        rospy.Subscriber('/current_velocity', TwistStamped, self.velocity_cb)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        self.pos_x = 0.
        self.pos_y = 0.
        self.pos_z = 0.
        self.current_orient = None
        self.yaw = 0.
        self.waypoints = None
        self.next_light = None
        self.velocity = 0.
        self.target_v = MAX_SPEED

        rospy.spin()

    def pose_cb(self, msg):
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

            if final_wpt_idx < len(self.waypoints):
                final_wpts.waypoints = self.waypoints[start_idx:final_wpt_idx]
            else:
                final_idx = (LOOKAHEAD_WPS + start_idx) % len(self.waypoints)
                final_wpts.waypoints = self.waypoints[start_idx:len(self.waypoints)] + self.waypoints[0:final_idx]

            # Acceleration
            v = self.velocity
            pose = Pose()
            pose.position = Point()
            pose.position.x = self.pos_x
            pose.position.y = self.pos_y
            for index in range(len(final_wpts.waypoints)):
                wpt = final_wpts.waypoints[index]
                new_v = min(math.sqrt(v ** 2 + 2 * self.max_acc * distance(pose, wpt.pose.pose)), MAX_SPEED)
                wpt.twist.twist.linear.x = MAX_SPEED
                v = new_v
                pose = wpt.pose.pose
            max_v = v

            # Deceleration
            if self.next_light is not None:
                s_light = distance(self.waypoints[start_idx].pose.pose, self.next_light.pose.pose)
                s_stop = s_light - STOP_LINE
                s_brake = s_stop - (-(v ** 2) / (2 * self.max_dec))

                for index in range(len(final_wpts.waypoints) - 1, -1, -1):
                    wpt = final_wpts.waypoints[index]
                    s_self = self.distance_wpt_to_car(wpt.pose.pose)
                    s = distance(pose, wpt.pose.pose)
                    if s_self > s_stop:
                        wpt.twist.twist.linear.x = 0
                    elif s_self > s_brake:
                        new_v = math.sqrt(max(v ** 2 + 2 * self.max_dec * s, 0))
                        wpt.twist.twist.linear.x = max_v - new_v
                        v = new_v
                    pose = wpt.pose.pose


            # elif DEBUG:
            #     print "Light is not found"
            s = "Waypoints: "
            for i in final_wpts.waypoints:
                s += str(i.twist.twist.linear.x) + " "
            print s
            self.final_waypoints_pub.publish(final_wpts)

            # if DEBUG:
            #     print "First waypoint is %d:%d" % (final_wpts.waypoints[0].pose.pose.position.x, final_wpts.waypoints[0].pose.pose.position.y)
            #     print "Car position is %d:%d" % (self.pos_x, self.pos_y)
            #     print "Final waypoints size = %d" % len(final_wpts.waypoints)

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

    def velocity_cb(self, msg):
        self.velocity = msg.twist.linear.x

    def traffic_lights_cb(self, msg):
        if self.current_orient is not None and self.waypoints is not None:
            lights = msg.lights
            light_dist = 1e10
            light_index = -1

            for idx in range(len(lights)):
                light = lights[idx]
                bearing = self.bearing_wpt_to_car(light.pose.pose)
                dist = self.distance_wpt_to_car(light.pose.pose)
                if dist < light_dist and bearing > 0.0:
                    light_index = idx
                    light_dist = dist

            if DEBUG:
                print "Distance to light %.15f :::: %d" % (light_dist, lights[light_index].state)
            if light_index >= 0 and lights[light_index].state != TrafficLight.GREEN:
                self.next_light = lights[light_index]
            else:
                self.next_light = None

                if DEBUG:
                    print "Light not found"

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def bearing_wpt_to_car(self, pose):
        shiftx = pose.position.x - self.pos_x
        shifty = pose.position.y - self.pos_y

        delta_x = shiftx * math.cos(self.yaw) - shifty * math.sin(self.yaw)
        delta_y = shiftx * math.sin(self.yaw) + shifty * math.cos(self.yaw)

        return math.atan2(delta_x, delta_y)

    def distance_wpt_to_car(self, pose):
        return math.sqrt((self.pos_x - pose.position.x) ** 2 +
                         (self.pos_y - pose.position.y) ** 2 +
                         (self.pos_z - pose.position.z) ** 2)

    def determine_next_wpt_idx(self):
        wpts = self.waypoints
        min_distance = 1e10
        index = 0

        for wpt_idx in range(len(wpts)):
            bearing = self.bearing_wpt_to_car(wpts[wpt_idx].pose.pose)
            dist = self.distance_wpt_to_car(wpts[wpt_idx].pose.pose)
            if dist < min_distance and bearing > 0.0:
                index = wpt_idx
                min_distance = dist

        return index


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
