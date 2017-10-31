import math
import sys

import numpy as np
import tensorflow as tf
from styx_msgs.msg import TrafficLight

from utils import label_map_util


def dist_box_center_to_point(box, point):
    box_center_y = (box[0] + box[2]) / 2
    box_center_x = (box[1] + box[3]) / 2
    return math.sqrt((point[0] - box_center_x)**2 + (point[1] - box_center_y)**2)


class TLClassifier(object):
    def __init__(self, path_to_ckpt, path_to_label_map, num_classes, score_threshold):
        self.score_threshold = score_threshold

        self.default_state = TrafficLight.UNKNOWN
        self.valid_classes = [1, 2, 3, 7]

        self.image_np_classified = None

        label_map = label_map_util.load_labelmap(path_to_label_map)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_classes, use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)

        self.detection_graph = tf.Graph()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            self.sess = tf.Session(graph=self.detection_graph, config=config)

        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

        print("Classifier graph loaded")

    def vote_on_states(self, states, boxes, point):
        result_state = self.default_state
        min_dist = sys.maxsize

        for i in range(len(boxes)):
            dist = dist_box_center_to_point(boxes[i], point)
            if dist < min_dist:
                min_dist = dist
                result_state = states[i]

        return result_state       

    def get_classification(self, image, projection_point):
        image_np_expanded = np.expand_dims(image, axis=0)

        with self.detection_graph.as_default():
            (boxes, scores, classes, num_detections) = self.sess.run(
                [self.detection_boxes, self.detection_scores, 
                self.detection_classes, self.num_detections],
                feed_dict={self.image_tensor: image_np_expanded})

        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes).astype(np.int32)
        result_state = self.default_state

        if num_detections[0] > 0:
            states = []
            boxes_filtered = []

            for i in range(boxes.shape[0]):
                if classes[i] not in self.valid_classes:
                    continue
                if scores[i] > self.score_threshold:
                    class_name = self.category_index[classes[i]]['name']
                    state = TrafficLight.UNKNOWN
                    if class_name == 'Red' or class_name == 'RED':
                        state = TrafficLight.RED
                    elif class_name == 'Yellow' or class_name == 'YELLOW':
                        state = TrafficLight.YELLOW
                    elif class_name == 'Green' or class_name == 'GREEN':
                        state = TrafficLight.GREEN

                    states.append(state)
                    boxes_filtered.append(boxes[i])
            
            if len(states) == 0:
                result_state = self.default_state
            elif len(states) == 1:
                result_state = states[0]
            elif states[1:] == states[:-1]:
                result_state = states[0]
            else:
                result_state = self.vote_on_states(states, boxes_filtered, projection_point)
        
        return result_state
