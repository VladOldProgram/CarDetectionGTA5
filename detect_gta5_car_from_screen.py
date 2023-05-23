import numpy as np
import tensorflow as tf
from grabscreen import grab_screen
import cv2
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


SAVED_MODEL_PATH = 'models/research/object_detection/efficientdet_d2_coco17_tpu-32/saved_model'
LABELMAP_PATH = 'training/labelmap.pbtxt'

labelmap = label_map_util.load_labelmap(LABELMAP_PATH)
categories = label_map_util.convert_label_map_to_categories(labelmap, max_num_classes=3, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

tf.keras.backend.clear_session()
saved_model = tf.saved_model.load(SAVED_MODEL_PATH)

while True:
    screen = cv2.resize(grab_screen(region=(0, 40, 800, 600)), (800, 600))
    image_np = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)

    image_np_expanded = np.expand_dims(image_np, axis=0)

    detections = saved_model(image_np_expanded)
    bboxes = detections['detection_boxes'][0].numpy()
    bclasses = detections['detection_classes'][0].numpy().astype(np.int32)
    bscores = detections['detection_scores'][0].numpy()

    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(bboxes),
        np.squeeze(bclasses),
        np.squeeze(bscores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=4,
        min_score_thresh=0.60
    )

    cv2.imshow('window', image_np)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break 