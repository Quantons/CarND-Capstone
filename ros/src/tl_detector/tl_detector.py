#!/usr/bin/env python
import glob
import math
import os
import sys

import numpy as np
import rospy
import tf
import yaml
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, Pose
from light_msgs.msg import UpcomingLight
from sensor_msgs.msg import Image
from std_msgs.msg import Int32
from styx_msgs.msg import Lane
from styx_msgs.msg import TrafficLightArray, TrafficLight

from light_classification.tl_classifier import TLClassifier

MODELS_PATH = os.path.dirname(os.path.realpath(__file__)) + '/light_classification/models'
MODEL_UDACITY = MODELS_PATH + '/graph_frcnn_resnet_real_udacity.pb'
LABELS_UDACITY = MODELS_PATH + '/label_map_udacity.pbtxt'
N_CLASSES_UDACITY = 4

MODEL_REAL = MODELS_PATH + '/graph_frcnn_resnet_sim_bosch.pb'
LABELS_REAL = MODELS_PATH + '/label_map_bosch.pbtxt'
N_CLASSES_REAL = 14

FX, FY = 1345.200806, 1353.838257
VISIBLE_DISTANCE = 200
SCORE_THRESHOLD = 0.5
STATE_COUNT_THRESHOLD = 3


def get_distance_between_poses(pose_1, pose_2):
    diff_x = pose_1.position.x - pose_2.position.x
    diff_y = pose_1.position.y - pose_2.position.y
    diff_z = pose_1.position.z - pose_2.position.z

    return math.sqrt(diff_x * diff_x + diff_y * diff_y + diff_z * diff_z)


def prepare_model_file(model_path):
    """Check if the model is in a single file or splitted in several files.
       If the file is splitted in several files, it creates a single file with
       all the parts in the right order.

    Args:
        model_path (String): model filename

    """
    if not os.path.exists(model_path):
        wildcard = model_path.replace('.pb', '.*')
        files = sorted([file for file in glob.glob(wildcard)])

        join_command = 'cat {} > {}'.format(" ".join(files), model_path)
        os.system(join_command)


def generate_upcominglight_msg(waypoint, id, pose, state):
    msg = UpcomingLight()
    msg.waypoint = waypoint
    msg.index = id
    msg.pose = pose
    msg.state = state
    return msg


def get_upcoming_stop_line_wp(car_position, all_stop_line_wps):
    interval = 0
    if car_position == 0:
        pass
    else:
        for i in range(len(all_stop_line_wps)):
            if car_position <= all_stop_line_wps[i]:
                interval = i
                break

    stop_line_wp = all_stop_line_wps[interval]
    light_id = interval

    return stop_line_wp, light_id


def get_unsqrt_distance_between_poses(pose_1, pose_2):
    diff_x = pose_1.position.x - pose_2.position.x
    diff_y = pose_1.position.y - pose_2.position.y

    return diff_x * diff_x + diff_y * diff_y


class TLDetector(object):
    def __init__(self):

        rospy.init_node('tl_detector')

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)
        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0
        self.lights = []
        self.all_stop_line_wps = None
        self.stop_line_positions = self.config['stop_line_positions']
        self.car_position = None
        self.is_running_simulator = False

        model = MODEL_UDACITY
        labels = LABELS_UDACITY
        n_classes = N_CLASSES_UDACITY
        if len(self.config['stop_line_positions']) > 1:
            model = MODEL_REAL
            labels = LABELS_REAL
            n_classes = N_CLASSES_REAL
            self.is_running_simulator = True

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        self.upcoming_light_pub = rospy.Publisher('/upcoming_light', UpcomingLight, queue_size=1)

        print "Start Tensorflow."
        self.bridge = CvBridge()

        prepare_model_file(model)
        self.light_classifier = TLClassifier(model, labels, n_classes, SCORE_THRESHOLD)
        self.light_classifier_on = True

        print "Tensorflow started."
        self.listener = tf.TransformListener()
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints
        if self.all_stop_line_wps is None and self.waypoints is not None:
            self.all_stop_line_wps = self.get_all_stop_line_wps(self.stop_line_positions)

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """

        self.has_image = True
        self.camera_image = msg

        stop_line_wp, state = self.process_traffic_lights()

        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            stop_line_wp = stop_line_wp if state == TrafficLight.RED else -1
            self.last_wp = stop_line_wp
        self.state_count += 1

    def get_closest_waypoint(self, pose):
        min_distance = sys.maxsize
        nearest_waypoint_index = -1

        if self.waypoints is not None:
            for i in range(0, len(self.waypoints.waypoints)):
                waypoint = self.waypoints.waypoints[i].pose.pose
                posepoint = pose
                distance = get_unsqrt_distance_between_poses(waypoint, posepoint)
                if distance < min_distance:
                    min_distance = distance
                    nearest_waypoint_index = i

        return nearest_waypoint_index

    def get_all_stop_line_wps(self, stop_line_positions):
        all_stop_line_wps = []
        pose = Pose()

        for i in range(len(stop_line_positions)):
            pose.position.x = stop_line_positions[i][0]
            pose.position.y = stop_line_positions[i][1]
            wp = self.get_closest_waypoint(pose)
            all_stop_line_wps.append(wp)
        return all_stop_line_wps

    def project_to_image_plane(self, point_in_world):
        image_width = self.config['camera_info']['image_width']
        image_height = self.config['camera_info']['image_height']
        x, y = None, None
        trans, rot = None, None
        try:
            now = rospy.Time.now()
            self.listener.waitForTransform("/base_link", "/world", now, rospy.Duration(1.0))
            (trans, rot) = self.listener.lookupTransform("/base_link", "/world", now)
        except (tf.Exception, tf.LookupException, tf.ConnectivityException):
            rospy.logerr("Failed to find camera to map transform")

        if trans is not None and rot is not None:
            transformation_matrix = self.listener.fromTranslationRotation(trans, rot)
            point_in_world_vector = np.array([[point_in_world.x], [point_in_world.y], [point_in_world.z], [1.0]], dtype=float)
            camera_point = np.dot(transformation_matrix, point_in_world_vector)
            x = int(-FX * camera_point[1] / camera_point[0] + image_width / 2)
            y = int(-FY * camera_point[2] / camera_point[0] + image_height / 2)

        return x, y

    def get_light_state(self, light):
        if not self.has_image:
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "rgb8")
        distance_car_tl = get_distance_between_poses(self.pose.pose, light.pose.pose)
        cv_image = self.crop_image(cv_image, distance_car_tl)

        if self.light_classifier_on is True:
            x = int(cv_image.shape[0] / 2)
            y = int(cv_image.shape[1] / 2)
            return self.light_classifier.get_classification(cv_image, (x, y))

        return TrafficLight.UNKNOWN

    def crop_image(self, image, distance):
        result = np.copy(image)
        print(distance)
        if self.is_running_simulator:
            if distance >= 150:
                top = 530
                bottom = 600
            elif distance >= 55:
                top = 340 + int((distance - 55.0) * ((530.0 - 340.0) / 95.0))
                bottom = 520 + int((distance - 55.0) * ((600.0 - 520.0) / 95.0))
            elif distance >= 27:
                top = 0 + int((distance - 27.0) * (340.0 / 28.0))
                bottom = 360 + int((distance - 27.0) * (520.0 - 360.0) / 28.0)
            else:
                top = 0
                bottom = 400

            result = result[top:bottom]

        else:
            result = result[0:740]

        return result

    def process_traffic_lights(self):
        light = None
        reaching_traffic_light = False
        light_id = -1
        stop_line_wp_index = -1

        if self.pose:
            closest_waypoint_to_car = self.get_closest_waypoint(self.pose.pose)
            if closest_waypoint_to_car != -1:
                self.car_position = closest_waypoint_to_car

        if self.all_stop_line_wps is not None and self.car_position is not None:
            stop_line_wp_index, light_id = get_upcoming_stop_line_wp(self.car_position, self.all_stop_line_wps)

            if self.pose is not None and stop_line_wp_index is not None:
                distance_to_stop_line = get_distance_between_poses(self.pose.pose, self.waypoints.waypoints[stop_line_wp_index].pose.pose)
                if distance_to_stop_line < VISIBLE_DISTANCE:
                    reaching_traffic_light = True
                    light = self.lights[light_id]

        if reaching_traffic_light:
            pred_state = self.get_light_state(light)
            upcoming_msg = generate_upcominglight_msg(stop_line_wp_index, light_id, self.lights[light_id].pose, pred_state)
            self.upcoming_light_pub.publish(upcoming_msg)
            return stop_line_wp_index, pred_state

        return -1, TrafficLight.UNKNOWN


if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
