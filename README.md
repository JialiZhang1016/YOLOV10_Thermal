# June 18 meeting

## 1. download yolov5

git clone https://github.com/ultralytics/yolov5.git
source venv/bin/activate
pip install -r requirements.txt

## 2. test the pre-trained model

```
python ./detect.py --source ./data/images --weights ./weight/yolov5n.pt --device mps
```

## 3. test the thermal_imaging_dataset

```
python ./detect.py --source ./datasets/thermal_imaging_dataset/video/thermal_8_bit --weights ./weight/yolov5n.pt --device mps
```

## 4. pre-trained model

https://github.com/ultralytics/yolov5/releases

## 5. lable your own dataset

pip install labelImg
labelIm

# June 25 meeting

## 1. test the video

video source: [highway vedio](https://www.bilibili.com/video/BV1AT411Q7eW/?spm_id_from=333.337.search-card.all.click&vd_source=d4a8071fb4c3dd2101835e7937300109)

```
python ./detect.py --source ./datasets/jiali_learn/videos/test_1.mp4 --weights ./weight/yolov5n.pt --view-img --device mps
```

## 2. test the computer camera

```
python ./detect.py --source 0 --weights ./weight/yolov5x.pt --view-img --device mps
```

## 3. difference between gray8 vs gray16

### RGB Images

- **Definition**: RGB stands for Red, Green, Blue. An RGB image is composed of three color channels, one for each primary color.
- **Color Representation**: Each pixel in an RGB image is represented by a combination of three values, one for each color channel (Red, Green, and Blue).
- **Bit Depth**: Typically, each color channel is represented with 8 bits, resulting in 24 bits per pixel (8 bits for Red, 8 bits for Green, and 8 bits for Blue). Each color has a single value ranging from 0 to 255.

### Gray8 Images

- **Definition**: A Gray8 image is a grayscale image where each pixel is represented by 8 bits.
- **Color Representation**: Each pixel has a single value ranging from 0 to 255, where 0 represents black, 255 represents white, and values in between represent varying shades of gray.
- **Bit Depth**: 8 bits per pixel.

### Gray16 Images

- **Definition**: A Gray16 image is a grayscale image where each pixel is represented by 16 bits.
- **Color Representation**: Each pixel has a single value ranging from 0 to 65535, allowing for a much finer gradation of shades of gray compared to Gray8.
- **Bit Depth**: 16 bits per pixel.

### Summary

- **RGB Images**: Full-color images with three 8-bit channels (24 bits per pixel, 3*2^16).
- **Gray8 Images**: Grayscale images with 8-bit depth (256 shades of gray, 2^8).
- **Gray16 Images**: Grayscale images with 16-bit depth (65536 shades of gray, 2^16) for higher precision and detail.

datasets/jiali_learn/gray8 vs gray16.py

## 4. run test thermal vedio folder and check the temperature

The way to edit [launch.json ](https://code.visualstudio.com/docs/python/debugging)

Detecting temperature from a grayscale 16-bit image, especially if it's a thermal image, involves understanding the specific properties of the image and the camera settings used to capture it. A grayscale 16-bit image can represent a wide range of values (from 0 to 65535), which can correspond to a range of temperatures if it's a thermal image.

**Linear Calibration Formula**:
A common linear formula is:

\[ T = a \cdot \text{PixelValue} + b \]

where:

- **T** is the temperature.
- **PixelValue** is the value of the pixel in the Gray16 image.
- **a** and **b** are calibration coefficients provided by the camera manufacturer.

# July 2 meeting

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

### 1.4 Final folder structure

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

You need to put the file in: `ultralytics/cfg/datasets/thermal_jiali.yaml`

From the official we know:

- Category 1:  People
- Category 2:  Bicycles -bicycles and motorcycles (not consistent with coco)
- Category 3:  Cars -personal vehicles and some small commercial vehicles.
- Category 17:  Dogs

But in the .yaml file, the category and name should aline with the .txt label file.

```
names:
  0: person
  1: bicycle
  2: car
  3: dogs
```

### 2.2 Train the model

Train the model using the python file `jiali_detect.py`

or

```
yolo detect train data=thermal_jiali.yaml model=ultralytics/cfg/models/v10/yolov10n.yaml epochs=10 batch=8 imgsz=640 device=0
```

You can change the `epoches`, `batch`, `device` based on your own training envirorment.

### 2.3 Validation

yolo val model=ultralytics/cfg/models/v10/yolov10n.yaml data=thermal_jiali.yaml batch=8

yolo val model=jameslahm/yolov10n data=coco8.yaml batch=8

yolo predict model=jameslahm/yolov10n

## Use app.py

change the `app.py` file

# July 7 meeting

## 1. Traning model on the Mill

### 1.1 Setup env in Mill

1. Upload zip file to the correct directory
2. Go to terminal and run: `unzip thermal_jiali.zip`
3. Clone the repo(public) to Mill: `git clone https://github.com/JialiZhang1016/YOLOV10_Thermal.git`
4. Create an virtual env
   ```
   module load anaconda
   conda create --prefix ./venv python=3.9 -y
   source activate ./venv
   pip install --upgrade pip
   pip install -r requirements.txt
   pip install -e .
   ```

### 1.2 Check the GPU and CPU

```
lscpu
nvidia-smi -q
```

### 1.3 Submit the job for training or val

```
sbatch yolov10_train.sub
```

## 2. Demo

### 2.1 Real-time detection using camera

```
yolo predict model=jameslahm/yolov10n source=0 conf=0.1 show=True

```

### 2.2 Real-time detection using thermal vedio

```
# Using official pretrained model 
yolo predict model=jameslahm/yolov10n source=path/to/thermal_imaging_video.mp4 conf=0.1 show=True

# Using our model
yolo predict model=path/to/best.pt source=path/to/thermal_imaging_video.mp4 conf=0.1 show=True
```

### 2.3 Demo in Gradio

We can use Gradio to create a demo to show our model. We just need to run

```
python app.py
```
