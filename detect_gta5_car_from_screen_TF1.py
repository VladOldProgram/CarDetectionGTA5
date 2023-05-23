import numpy as np
import tensorflow as tf
from grabscreen import get_screenshot
import cv2
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
'''
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

boxes_centres_x = []
boxes_centres_y = []
t = 0
boxes_centres_z = []
'''
SAVED_MODEL_PATH = 'models/research/object_detection/ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.pb'
LABELMAP_PATH = 'models/research/object_detection/data/mscoco_label_map.pbtxt'
#VIDEO_PATH = 'videos/road2.mp4'

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(SAVED_MODEL_PATH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

labelmap = label_map_util.load_labelmap(LABELMAP_PATH)
categories = label_map_util.convert_label_map_to_categories(labelmap, max_num_classes=90, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
threshold = 0.4
critical_distance = 0.5
#class_id = 3

'''
def ExtractBoxes(boxes, classes, scores, image_width, image_height):
    extracted_bboxes = []
    for i in range(len(boxes)):
        if (classes[i] == class_id):
            if (scores[i] >= threshold):
                xmin = int(boxes[i][1] * image_width)
                ymin = int(boxes[i][0] * image_height)
                xmax = int(boxes[i][3] * image_width)
                ymax = int(boxes[i][2] * image_height)
                boxes_centres_x.append(int(xmin + (xmax - xmin) / 2))
                boxes_centres_y.append(int(ymin + (ymax - ymin) / 2))
                global t
                t += 1
                boxes_centres_z.append(t)

                class_label = category_index[int(classes[i])]['name']
                extracted_bboxes.append([xmin, ymin, xmax, ymax, class_label, float(scores[i])])

    return extracted_bboxes


def ShowDetections(image, boxes):
    for i in range(len(boxes)):
        xmin = boxes[i][0]
        ymin = boxes[i][1]
        xmax = boxes[i][2]
        ymax = boxes[i][3]
        #object_class = str(boxes[i][4])
        #score = str(np.round(boxes[i][-1], 2))

        #text = object_class + ': ' + score
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
        #cv2.putText(image, text, (xmin + 5, ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return image
'''

#capture = cv2.VideoCapture(VIDEO_PATH)
with detection_graph.as_default():
    with tf.compat.v1.Session(graph=detection_graph) as sess:
        image_width = 800
        image_height = 600
        '''
        object_trace = []
        f = 0
        c = 5
        '''
        while True:
        #while capture.isOpened():
            '''
            ret, image = capture.read()
            if not ret:
                break
            image_height, image_width, _ = image.shape
            '''
            screenshot = cv2.resize(get_screenshot((0, 40, image_width, image_height)), (image_width, image_height))
            image = cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB)
            
            image_expanded = np.expand_dims(image, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_expanded})
            '''
            if (f % c == 0) :
                object_trace.extend(ExtractBoxes(boxes[0], classes[0], scores[0], image_width, image_height))
            f += 1

            if (len(boxes) != 0):
               image = ShowDetections(image, object_trace)
            '''
            vis_util.visualize_boxes_and_labels_on_image_array(
                image,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=4,
                min_score_thresh=threshold
            )

            for i, unused in enumerate(boxes[0]):
                if classes[0][i] == 3 or classes[0][i] == 6 or classes[0][i] == 8:
                    if scores[0][i] >= threshold:
                        middle_x = (boxes[0][i][1] + boxes[0][i][3]) / 2
                        middle_y = (boxes[0][i][0] + boxes[0][i][2]) / 2
                        distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1])) ** 4), 1)
                        if distance <= critical_distance:
                            if middle_x > 0.3 and middle_x < 0.7:
                                cv2.putText(image, 'Warning!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                                break
            
            cv2.imshow('Detection', image)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break 
        #capture.release()
'''
plt.rcParams['figure.figsize'] = (6, 6)
fig = plt.figure().add_subplot(projection='3d')

fig.plot(boxes_centres_x, boxes_centres_y, boxes_centres_z)

fig.set_xlabel('x')
fig.set_ylabel('y')
fig.set_zlabel('z')
fig.set_xlim(0, 1920)
fig.set_ylim(0, 1080)

plt.show()
'''