model_dir=D:/VSProjects/CarDetectionGTA5/models/research/object_detection/faster_rcnn_resnet50_v1_1024x1024_coco17_GTA5_tpu-8/
out_dir=$model_dir/exported_model
mkdir -p $out_dir

python D:/VSProjects/CarDetectionGTA5/models/research/object_detection/exporter_main_v2.py \
    --input_type="image_tensor" \
    --pipeline_config_path=D:/VSProjects/CarDetectionGTA5/training/faster_rcnn_resnet50_v1_1024x1024_coco17_tpu-8.config \
    --trained_checkpoint_dir=$model_dir \
    --output_directory=$out_dir