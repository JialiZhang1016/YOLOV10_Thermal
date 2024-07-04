## 1. Convert the COCO format to YOLO format

### 1.1 COCO format

```
<dataset_dir>/
    images/
        <imagename0>.<ext>
        <imagename1>.<ext>
        <imagename2>.<ext>
        ...
    annotations/
        train_labels.json
        val_labels.json
        test_labels.json
```

### 1.2 YOLO format

see this link: https://docs.ultralytics.com/datasets/detect/#ultralytics-yolo-format

```
.
├── LICENSE
├── README.md
├── images
│   ├── train
│   │   ├── 000000000009.jpg
│   │   ├── 000000000025.jpg
│   │   ├── 000000000030.jpg
│   │   └── 000000000034.jpg
│   └── val
│       ├── 000000000036.jpg
│       ├── 000000000042.jpg
│       ├── 000000000049.jpg
│       └── 000000000061.jpg
└── labels
    ├── train
    │   ├── 000000000009.txt
    │   ├── 000000000025.txt
    │   ├── 000000000030.txt
    │   └── 000000000034.txt
    └── val
        ├── 000000000036.txt
        ├── 000000000042.txt
        ├── 000000000049.txt
        └── 000000000061.txt
```

### 1.3 How to convert it

Use `ultralytics/data/converter.py`

But before you run it, you have add these 2 lines before you run the code. It will make sure you have the directory to write the lable file.

```
            # Ensure the output directory exists
            output_dir = (fn / f).parent
            output_dir.mkdir(parents=True, exist_ok=True)
  
            # Write
            output_file = (fn / f).with_suffix(".txt")
```

Then we can use this python script `jiali_converter_to_yolo.py` to convert.

There is a issue after convert: the lable of the `dog` is 15. We need to change it to be 3.

Here is the code to change the lable: `/Users/captainzhang/Desktop/Yolo/datasets/15to3.py`

Reference:

- https://docs.ultralytics.com/datasets/detect/#port-or-convert-label-formats
- https://docs.ultralytics.com/reference/data/converter/#ultralytics.data.converter.coco80_to_coco91_class

### 1.4 Final folder

```
(venv) captainzhang@Captains-MacBook-Air thermal_jiali % tree -L 2
.
├── images
│   ├── train
│   └── val
└── labels
    ├── train
    └── val
```

## 2. Train our model

### 2.1 Create the .yaml file

ultralytics/cfg/datasets/thermal_jiali.yaml

yolo val model=jameslahm/yolov10n data=thermal_jiali.yaml batch=8

- Category 1:  People
- Category 2:  Bicycles -bicycles and motorcycles (not consistent with coco)
- Category 3:  Cars -personal vehicles and some small commercial vehicles.
- Category 17:  Dogs

### 2.2 Train the model

Train the model `jiali_detect.py`
or

yolo detect train data=thermal_jiali.yaml model=ultralytics/cfg/models/v10/yolov10n.yaml epochs=10 batch=8 imgsz=640 device=cpu

### 2.3 Validation

yolo val model=ultralytics/cfg/models/v10/yolov10n.yaml data=thermal_jiali.yaml batch=8

yolo val model=jameslahm/yolov10n data=coco8.yaml batch=8

yolo predict model=jameslahm/yolov10n

## Use app.py

change the `app.py` file
