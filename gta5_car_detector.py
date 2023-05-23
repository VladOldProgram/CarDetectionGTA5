import cv2
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util


class GTA5CarDetector:

    def __init__(self, path_to_checkpoint: str, path_to_labelmap: str, class_id=1, threshold=0.5):
        self.class_id = class_id
        self.threshold = threshold

        label_map = label_map_util.load_labelmap(path_to_labelmap)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=3, use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)
        
        tf.keras.backend.clear_session()
        self.saved_model = tf.saved_model.load(path_to_checkpoint)


    def ExtractBBoxes(self, bboxes, bclasses, bscores, image_width, image_height):
        extracted_bboxes = []
        for i in range(len(bboxes)):
            if (self.class_id is None) or (bclasses[i] in self.class_id):
                if (bscores[i] >= self.threshold):
                    xmin = int(bboxes[i][1] * image_width)
                    ymin = int(bboxes[i][0] * image_height)
                    xmax = int(bboxes[i][3] * image_width)
                    ymax = int(bboxes[i][2] * image_height)

                    class_label = self.category_index[int(bclasses[i])]['name']
                    extracted_bboxes.append([xmin, ymin, xmax, ymax, class_label, float(bscores[i])])

        return extracted_bboxes


    def DetectFromImage(self, image):
        image_height, image_width, _ = image.shape
        input_tensor = np.expand_dims(image, 0)

        detections = self.saved_model(input_tensor)
        bboxes = detections['detection_boxes'][0].numpy()
        bclasses = detections['detection_classes'][0].numpy().astype(np.int32)
        bscores = detections['detection_scores'][0].numpy()

        detection_boxes = self.ExtractBBoxes(bboxes, bclasses, bscores, image_width, image_height)

        return detection_boxes


    def ShowDetections(self, image, boxes, detection_time=None):
        if (len(boxes) == 0):
            return image

        for i in range(len(boxes)):
            xmin = boxes[i][0]
            ymin = boxes[i][1]
            xmax = boxes[i][2]
            ymax = boxes[i][3]
            object_class = str(boxes[i][4])
            score = str(np.round(boxes[i][-1], 2))

            text = object_class + ': ' + score
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
            cv2.putText(image, text, (xmin + 5, ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        if detection_time != None:
            fps = str(round(1000.0 / detection_time, 1)) + " FPS"
            cv2.putText(image, fps, (25, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        return image
