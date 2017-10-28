#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from tl_ssd_detector.tl_ssd_detector import TLSSDDetector
from tl_cnn_classifier.tl_cnn_classifier import TLCNNClassifier

import tf
import cv2
import yaml
import math

STATE_COUNT_THRESHOLD = 2
DEBUG = False


def distance(x1, y1, x2, y2):
    return math.sqrt(((x2 - x1) ** 2) + ((y2 - y1) ** 2))

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.image_index = 0

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb, queue_size=1)
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb, queue_size=1)
        rospy.Subscriber('/image_color', Image, self.image_cb, queue_size=1)

        self.upcoming_traffic_light_wp_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)
        self.upcoming_traffic_light_state_pub = rospy.Publisher('/traffic_state', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLCNNClassifier()
        self.listener = tf.TransformListener()
        self.tl_detector = TLSSDDetector()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints
        self.sub2.unregister()

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.image_index += 1
        if self.image_index % 5 != 1:
            # skip 4 images out of 5
            return

        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()

        self.upcoming_traffic_light_wp_pub.publish(Int32(light_wp))

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_traffic_light_state_pub.publish(Int32(self.state))
        else:
            self.upcoming_traffic_light_state_pub.publish(Int32(self.last_state))
        self.state_count += 1

    def find_closest_wpt(self, position):
        if self.waypoints is not None:
            closest_distance = float('inf')
            closest_id = 0
            x = position[0]
            y = position[1]

            for i, waypoint in enumerate(self.waypoints.waypoints):
                dist = distance(x, y, waypoint.pose.pose.position.x, waypoint.pose.pose.position.y)
                if dist < closest_distance:
                    closest_id = i
                    closest_distance = dist

            return closest_id

    def get_closest_waypoint(self, pose):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        return self.find_closest_wpt([pose.position.x, pose.position.y])

    def get_light_state(self):
        """Determines the current color of the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if not self.has_image:
            return False

        tl_state = TrafficLight.UNKNOWN

        image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bbox = self.tl_detector.get_strongest_bbox(image)

        if bbox is not None:
            ymin = int(bbox[0] * image.shape[0])
            xmin = int(bbox[1] * image.shape[1])
            ymax = int(bbox[2] * image.shape[0])
            xmax = int(bbox[3] * image.shape[1])
            cropped_image = image[ymin:ymax, xmin:xmax, :]
            resized_image = cv2.resize(cropped_image, (24, 72))

            tl_state = self.light_classifier.get_classification(resized_image)
            if tl_state != TrafficLight.GREEN:
                tl_state = TrafficLight.RED

        return tl_state

    def get_closest_traffic_light(self, pose):
        traffic_light_positions = self.config['stop_line_positions']
        closest_distance = float('inf')
        index = -1
        for light_position in traffic_light_positions:
            index += 1
            dist = distance(pose.position.x, pose.position.y, light_position[0], light_position[1])
            if dist < closest_distance:
                closest_id = index
                closest_distance = dist
                closest_light_wp = light_position
        return closest_distance, closest_id, closest_light_wp

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        light_wp, state = -1, TrafficLight.UNKNOWN
        car_position = None
        stop_line_wp_index = -1

        if self.pose:
            car_position = self.get_closest_waypoint(self.pose.pose)

        if car_position:
            closest_light_dist, closest_light_id, light_wp = self.get_closest_traffic_light(self.pose.pose)

        if DEBUG:
            rospy.loginfo("Current vehicle position: [x:{}, y:{}, z:{}]".format(self.pose.pose.position.x, self.pose.pose.position.y, self.pose.pose.position.z))
            rospy.loginfo("Closest traffic light position: {}".format(light_wp))
            rospy.loginfo("Closest traffic light index: {}".format(self.find_closest_wpt(light_wp)))
            rospy.loginfo("Current vehicle way point: {}".format(car_position))
            rospy.loginfo("Closest traffic light way point distance: {}".format(closest_light_dist))

        if closest_light_dist < 150:
            state = self.get_light_state()
            stop_line_wp_index = self.find_closest_wpt(light_wp)

        return stop_line_wp_index, state

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
