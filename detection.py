import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2
import subprocess
import socket
import datetime
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


class TensorObject:
    def __init__(self, image_np, detection_graph):
        self.image_np_expanded = np.expand_dims(image_np, axis=0)
        self.image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        self.boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        self.scores = detection_graph.get_tensor_by_name('detection_scores:0')
        self.classes = detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = detection_graph.get_tensor_by_name('num_detections:0')


MODEL_NAME = 'ssd_inception_v2_coco_2017_11_17'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 90

cap = cv2.VideoCapture(0)
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def download_model():
    opener = urllib.request.URLopener()
    opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
    tar_file = tarfile.open(MODEL_FILE)
    for file in tar_file.getmembers():
        file_name = os.path.basename(file.name)
        if 'frozen_inference_graph.pb' in file_name:
            tar_file.extract(file, os.getcwd())


def check_model():
    if os.path.exists(MODEL_NAME) is False:
        print("Downloading pre-trained model %s..." % MODEL_NAME)
        download_model()


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def create_graph():
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph


def get_detected_objects(scores, classes, boxes, limit):
    objects = []
    nbr = scores.shape[1]
    for j in range(nbr):
        object_id = int(classes[0][j])
        object_name = category_index[object_id]["name"]
        object_score = scores[0][j]
        if (object_score >= limit):
            objects.append(object_name)
    return objects


def get_object_after_run(image_np, detection_graph, sess):
    t_object = TensorObject(image_np, detection_graph)

    (t_object.boxes, t_object.scores, t_object.classes,
        t_object.num_detections) = sess.run([t_object.boxes,
        t_object.scores, t_object.classes, t_object.num_detections],
        feed_dict={t_object.image_tensor: t_object.image_np_expanded})
    return t_object


def write_data(time, data):
    with open("created_data/data.txt", "a") as text_file:
        text_file.write(time + "_" + str(data) + "\n")


def count_persons(image_id, image_np, t_object):
    objects = get_detected_objects(t_object.scores, t_object.classes, 
        t_object.boxes, 0.5)
    image_id[0] += 1
    if image_id[0] % 5 == 0:
        write_data(str(datetime.datetime.now()), objects.count("person"))


def launch_loop_detection(detection_graph, sess):
    image_id = [0]
    while True:
        ret, image_np = cap.read()
        if ret is True:
            t_object = get_object_after_run(image_np, detection_graph, sess)
            count_persons(image_id, image_np, t_object)
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(t_object.boxes),
                np.squeeze(t_object.classes).astype(np.int32),
                np.squeeze(t_object.scores),
                category_index,
                use_normalized_coordinates=True,
                min_score_thresh=.5,
                line_thickness=4)
            cv2.imshow('Detection', cv2.resize(image_np, (800, 600)))
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break


def proceed_to_person_detection():
    detection_graph = create_graph()

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            launch_loop_detection(detection_graph, sess)


def main():
    if len(sys.argv) > 1:
        if sys.argv[1] == "-d":
            os.remove("created_data/data.txt")
    check_model()
    proceed_to_person_detection()


if __name__ == "__main__":
    main()
