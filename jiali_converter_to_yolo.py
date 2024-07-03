# converter.py file content should be present here as provided in your input

from ultralytics.data.converter import convert_coco

convert_coco(labels_dir="ultralytics/datasets/thermal_imaging_dataset/images/annotations", use_segments=False, use_keypoints=False)