out_dir=D:/VSProjects/CarDetectionGTA5/models/research/object_detection/faster_rcnn_resnet50_v1_1024x1024_coco17_GTA5_tpu-8
mkdir -p $out_dir
python D:/VSProjects/CarDetectionGTA5/models/research/object_detection/model_main_tf2.py --alsologtostderr --model_dir=$out_dir --checkpoint_every_n=500  \
                         --pipeline_config_path=D:/VSProjects/CarDetectionGTA5/training/faster_rcnn_resnet50_v1_1024x1024_coco17_tpu-8.config \
                         --eval_on_train_data 2>&1 | tee $out_dir/train.log
