export CUDA_VISIBLE_DEVICES="-1"

out_dir=D:/VSProjects/CarDetectionGTA5/models/research/object_detection/faster_rcnn_resnet50_v1_1024x1024_coco17_GTA5_tpu-8/
mkdir -p $out_dir
python D:/VSProjects/CarDetectionGTA5/models/research/object_detection/model_main_tf2.py --alsologtostderr --model_dir=$out_dir \
                         --pipeline_config_path=D:/VSProjects/CarDetectionGTA5/training/faster_rcnn_resnet50_v1_1024x1024_coco17_tpu-8.config \
                         --checkpoint_dir=$out_dir 2>&1 | tee $out_dir/eval.log
