from os import path

import numpy as np
import tensorflow as tf

from detection_unit.redis_broker import BaseModelProcessor
from detection_unit.redis_broker import get_logger
from utils import label_map_util
from utils.utils import load_image_into_numpy_array
from config import config


logger = get_logger(__name__)


basepath = path.dirname(__file__)

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = path.join(basepath, 'model_data', 'mscoco_label_map.pbtxt')
GRAPH_PATH = path.join(basepath, 'model_data', 'frozen_inference_graph.pb')

NUM_CLASSES = 90

MIN_SCORE_THRESH = float(config.get('model', 'detection_threshold'))
MIN_AREA_THRESH = float(config.get('model', 'min_area_threshold'))
MAX_AREA_THRESH = float(config.get('model', 'max_area_threshold'))
OVERLAP_THRESH = float(config.get('model', 'overlap_threshold'))


tf_config = tf.ConfigProto(allow_soft_placement=True)
tf_config.gpu_options.allocator_type = 'BFC'
tf_config.gpu_options.per_process_gpu_memory_fraction = 0.2


class ObjectDetector(BaseModelProcessor):
    def __init__(self):
        """
            Builds Tensorflow graph, load model and labels
        """
        self.classes_to_detect = ['person']
        # Load lebel_map
        self._load_label(PATH_TO_LABELS, NUM_CLASSES, use_disp_name=True)

        # Load Tensorflow model into memory
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(GRAPH_PATH, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        with self.detection_graph.as_default():
            self.sess = tf.Session(graph=self.detection_graph, config=tf_config)
            # Definite input and output Tensors for detection_graph
            self.image_tensor = self.detection_graph.get_tensor_by_name(
                'image_tensor:0')
            # Each box represents a part of the image where a particular
            # object was detected.
            self.detection_boxes = self.detection_graph.get_tensor_by_name(
                'detection_boxes:0')
            # Each score represent how level of confidence for each of
            # the objects. Score is shown on the result image, together
            # with the class label.
            self.detection_scores = self.detection_graph.get_tensor_by_name(
                'detection_scores:0')
            self.detection_classes = self.detection_graph.get_tensor_by_name(
                'detection_classes:0')
            self.num_detections = self.detection_graph.get_tensor_by_name(
                'num_detections:0')

        logger.info('Model graph loaded.')

    def process(self, message):
        frame_url = message['path_to_img']

        frame = load_image_into_numpy_array(frame_url)
        boxes, scores = self._detect_person(frame, MIN_SCORE_THRESH)

        boxes, scores = self._filter_bb_by_size(
            boxes, scores, frame, MIN_AREA_THRESH, MAX_AREA_THRESH)
        boxes, scores = self._non_max_suppression_fast(
            boxes, scores, OVERLAP_THRESH)

        logger.debug('boxes: {}.\nscores {}'.format(boxes, scores))

        response = message
        response.update({'boxes': boxes, 'scores': scores})
        return response

    def _detect_person(self, frame, threshold=0.6):
        """
           Predicts person in frame with threshold level of confidence
           Returns list with top-left, bottom-right coordinates and
           list with labels, confidence in %
        """
        frames = np.expand_dims(frame, axis=0)
        # Actual detection.
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores,
             self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: frames})

        # Find detected boxes coordinates
        return self._boxes_coordinates(
            frame,
            boxes[0],
            classes[0].astype(np.int32),
            scores[0],
            min_score_thresh=threshold
        )

    def _boxes_coordinates(self,
                           image,
                           boxes,
                           classes,
                           scores,
                           max_boxes_to_draw=20,
                           min_score_thresh=.5):
        """
          This function groups boxes that correspond to the same location
          and creates a display string for each detection and overlays these
          on the image.

          Args:
            image: uint8 numpy array with shape (img_height, img_width, 3)
            boxes: a numpy array of shape [N, 4]
            classes: a numpy array of shape [N]
            scores: a numpy array of shape [N] or None.  If scores=None, then
              this function assumes that the boxes to be plotted are groundtruth
              boxes and plot all boxes as black with no classes or scores.
            category_index: a dict containing category dictionaries (each holding
              category index `id` and category name `name`) keyed by category indices.
            use_normalized_coordinates: whether boxes is to be interpreted as
              normalized coordinates or not.
            max_boxes_to_draw: maximum number of boxes to visualize.  If None, draw
              all boxes.
            min_score_thresh: minimum score threshold for a box to be visualized
        """

        if not max_boxes_to_draw:
            max_boxes_to_draw = boxes.shape[0]
        number_boxes = min(max_boxes_to_draw, boxes.shape[0])
        final_boxes = []
        final_scores = []
        for i in range(number_boxes):
            if self.category_index[classes[i]]['name'] not in \
                    self.classes_to_detect:
                continue
            if scores is None or scores[i] > min_score_thresh:
                box = tuple(boxes[i].tolist())
                ymin, xmin, ymax, xmax = box

                im_height, im_width, _ = image.shape
                left, right, top, bottom = [int(z) for z in
                                            (xmin * im_width, xmax * im_width,
                                             ymin * im_height,
                                             ymax * im_height)]

                final_boxes.append([top, left, bottom, right])
                final_scores.append(scores[i])
        return final_boxes, final_scores

    def _load_label(self, path, num_c, use_disp_name=True):
        """
            Loads labels
        """
        label_map = label_map_util.load_labelmap(path)
        categories = label_map_util.convert_label_map_to_categories(
            label_map, max_num_classes=num_c, use_display_name=use_disp_name)
        self.category_index = label_map_util.create_category_index(categories)

    @staticmethod
    def _filter_bb_by_size(boxes, scores, frame, min_area_thresh=0.005,
                           max_area_thresh=0.25):
        h, w, _ = frame.shape
        frame_area = h * w
        res_bbs = []
        res_labels = []
        for box, score in zip(boxes, scores):
            s = (box[2] - box[0]) * (box[3] - box[1])
            if frame_area * min_area_thresh < s < frame_area * max_area_thresh:
                res_bbs.append(box)
                res_labels.append(score)

        return res_bbs, res_labels

    @staticmethod
    def _non_max_suppression_fast(boxes, scores, overlap_thresh=0.35):
        # if there are no boxes, return an empty list
        boxes = np.array(boxes)
        if len(boxes) == 0:
            return [], []

        # initialize the list of picked indexes
        pick = []

        # grab the coordinates of the bounding boxes
        y1 = boxes[:, 0]
        x1 = boxes[:, 1]
        y2 = boxes[:, 2]
        x2 = boxes[:, 3]

        # compute the area of the bounding boxes and sort the bounding
        # boxes by the bottom-right y-coordinate of the bounding box
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)

        # keep looping while some indexes still remain in the indexes
        # list
        while len(idxs) > 0:
            # grab the last index in the indexes list and add the
            # index value to the list of picked indexes
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            # find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            # compute the width and height of the bounding box
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            # compute the ratio of overlap
            overlap = 1. * (w * h) / area[idxs[:last]]

            # delete all indexes from the index list that have
            idxs = np.delete(idxs, np.concatenate(
                ([last], np.where(overlap > overlap_thresh)[0])))

        # return only the bounding boxes that were picked using the
        # integer data type
        return boxes[pick], [scores[i] for i in pick]
