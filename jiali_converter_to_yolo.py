# converter.py file content should be present here as provided in your input
from ultralytics.data.converter import convert_coco

labels_dir="/Users/captainzhang/Desktop/Yolo/datasets/thermal_imaging_dataset/images/annotations"
save_dir="/Users/captainzhang/Desktop/Yolo/datasets/thermal_coco/"

convert_coco(labels_dir, save_dir, use_segments=False, use_keypoints=False, cls91to80=True)
