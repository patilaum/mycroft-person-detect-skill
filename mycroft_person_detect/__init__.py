# Copyright 2016 Mycroft AI, Inc.
#
# This file is part of Mycroft Core.
#
# Mycroft Core is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Mycroft Core is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Mycroft Core.  If not, see <http://www.gnu.org/licenses/>.

from adapt.intent import IntentBuilder

from mycroft.skills.core import MycroftSkill
from mycroft.util.log import getLogger

__author__ = 'patilaum'
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

import cv2
from matplotlib import pyplot as plt
import time

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

get_ipython().run_line_magic('matplotlib', 'inline')

import object_detection

MODEL_NAME = 'detection_graph/'

PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

LOGGER = getLogger(__name__)


class PersonDetectSkill(MycroftSkill):
    def __init__(self):
        super(PersonDetectSkill, self).__init__(name="PersonDetectSkill")

    def initialize(self):
        how_many_intent = IntentBuilder("HowManyIntent"). \
            require("HowManyKeyword").build()
        self.register_intent(how_many_intent, self.handle_how_many_intent)
        
        
    def handle_how_many_intent(self, message):
        capture=cv2.VideoCapture(0)
        capture.set(3,640)
        capture.set(4,480)
        frame_set=[]
        start_time=time.time()
        while(True):
            ret, frame = capture.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_set.append(gray)
            ##cv2.imshow('frame',gray)
            
            end_time=time.time()
            elapsed = end_time - start_time
            if elapsed > 2:
               break
        capture.release()
        with detection_graph.as_default():
          with tf.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            ##for image in frame_set:
            ##image_path = frame_set[0]
            ##image = Image.open(image_path)
          # the array based representation of the image will be used later in order to prepare the
          # result image with boxes and labels on it.
            image_np = frame_set[-1]
          # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
          # Actual detection.
            (boxes, scores, classes, num) = sess.run(
              [detection_boxes, detection_scores, detection_classes, num_detections],
              feed_dict={image_tensor: image_np_expanded})
          # Visualization of the results of a detection.
            width, height = image.size
            final_score = np.squeeze(scores)    
            count = 0
            for i in range(100):
                if scores is None or final_score[i] > 0.5:
                        count = count + 1
                        
            
                    
       
        
        if count > 1:
            self.speak_dialog("There are {} persons in front of me" .format(count))
        if count == 0:
            self.speak_dialog("I cant see anyone")
        if count == 1:
            self.speak_dialog("There is one person in front of me")
        
    
    def stop(self):
        pass


def create_skill():
    return PersonDetectSkill()
