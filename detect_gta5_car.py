import cv2
import os
import time
import argparse
from gta5_car_detector import GTA5CarDetector


def DetectGTA5CarFromVideo(gta5_car_detector, path_to_video: str, should_save_output=False, output_directory='output/'):
    capture = cv2.VideoCapture(path_to_video)
    if should_save_output:
        output_path = os.path.join(output_directory, 'detection_' + path_to_video.split('/')[-1])
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

    while capture.isOpened():
        ret, image = capture.read()
        if not ret:
            break

        start_time = time.time()
        detection_boxes = gta5_car_detector.DetectFromImage(image)
        elapsed_time = round((time.time() - start_time) * 1000)
        image = gta5_car_detector.ShowDetections(image, detection_boxes, detection_time=elapsed_time)

        cv2.imshow('GTA5 car detection', image)
        if cv2.waitKey(1) == 27:
            break

        if should_save_output:
            video_writer.write(image)

    capture.release()
    if should_save_output:
        video_writer.release()


def DetectGTA5CarFromWebcam():
    pass


def DetectGTA5CarFromImages(detector, images_directory, should_save_output=False, output_directory='output/'):
    for file in os.scandir(images_directory):
        if file.is_file() and file.name.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(images_directory, file.name)
            print(image_path)
            image = cv2.imread(image_path)

            detection_boxes = detector.DetectFromImage(image)
            image = detector.ShowDetections(image, detection_boxes)
            cv2.imshow('GTA5 car detection', image)
            cv2.waitKey(0)

            if should_save_output:
                output_image_path = os.path.join(output_directory, file.name)
                cv2.imwrite(output_image_path, image)


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser(description='GTA5 car detection from images, video or webcam')
    argument_parser.add_argument(
        '--model_path', 
        help='Path to frozen detection model', 
        default='models/research/object_detection/efficientdet_d2_coco17_tpu-32/saved_model'
    )
    argument_parser.add_argument(
        '--path_to_labelmap', 
        help='Path to labelmap (.pbtxt) file', 
        default='training/labelmap.pbtxt'
    )
    argument_parser.add_argument(
        '--class_ids',
        help='id of classes to detect, expects string with ids delimited by ","',
        type=str, 
        default='3') # в документе должно быть '1'
    argument_parser.add_argument(
        '--threshold', 
        help='Detection threshold', 
        type=float, 
        default=0.4
    )
    argument_parser.add_argument(
        '--images_dir', 
        help='Directory to input images)', 
        default='images/test'
    )
    argument_parser.add_argument(
        '--video_path', 
        help='Path to input video)', 
        default='videos/video1.mp4'
    )
    argument_parser.add_argument(
        '--output_directory', 
        help='Path to output images and video', 
        default='output'
    )
    argument_parser.add_argument(
        '--video_input', 
        help='Flag for video input, default: False', 
        action='store_true'
    )
    argument_parser.add_argument(
        '--webcam_input', 
        help='Flag for webcam input, default: False', 
        action='store_true'
    )
    argument_parser.add_argument(
        '--save_output', 
        help='Flag for save images and video with detections visualized, default: False',        
        action='store_true'
    )
    args = argument_parser.parse_args()

    class_id = None
    if args.class_ids is not None:
        class_id = [int(item) for item in args.class_ids.split(',')]

    if args.save_output:
        if not os.path.exists(args.output_directory):
            os.makedirs(args.output_directory)

    detector = GTA5CarDetector(args.model_path, args.path_to_labelmap, class_id, args.threshold)

    if args.video_input:
        DetectGTA5CarFromVideo(detector, args.video_path, args.save_output, args.output_directory)
    elif args.webcam_input:
        DetectGTA5CarFromWebcam()
    else:
        DetectGTA5CarFromImages(detector, args.images_dir, args.save_output, args.output_directory)

    print('Done...')
    cv2.destroyAllWindows()
