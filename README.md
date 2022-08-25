# YOLOv7 based PPE Detector

Thanks to the autors of paper - [YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors](https://arxiv.org/abs/2207.02696)

## To perform inference
extract the best.7z.001 weights from runs/train/ppedetector and then use the below command in anaconda promt.

python detectppe.py --weights runs/train/ppedetector/best.pt --conf 0.5 --source ./test_video/test.mp4

If you want to test on ip feed you just have to replace the source path with the ip address. Results will be store into "runs/detect/"

## To train on a custom dataset
Download your custom dataset and paste into main repository and use the below command to train yolov7 on your own dataset.
python train.py --batch=4 --cfg=cfg/training/yolov7.yaml --epochs=300 --data=./dataset/data.yaml --weights='./yolov7.pt' --device=0

its better to use GPU for training and the "device" should be n-1 which means n=number of GPUs, in my case I used so therefore it is "device=0"

## Results

<img src="https://github.com/sharjeelanjum/PPE_Detector_YOLOV7/blob/main/runs/train/ppedetector/results.png" width="400" />

<img src="https://github.com/sharjeelanjum/PPE_Detector_YOLOV7/blob/main/runs/train/ppedetector/%24filename003.jpg" width="400" />
<img src="https://github.com/sharjeelanjum/PPE_Detector_YOLOV7/blob/main/runs/train/ppedetector/%24filename010.jpg" width="400" />
<img src="https://github.com/sharjeelanjum/PPE_Detector_YOLOV7/blob/main/runs/train/ppedetector/%24filename064.jpg" width="400" />


## Citation

```
@article{wang2022yolov7,
  title={{YOLOv7}: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors},
  author={Wang, Chien-Yao and Bochkovskiy, Alexey and Liao, Hong-Yuan Mark},
  journal={arXiv preprint arXiv:2207.02696},
  year={2022}
}
```
```
@misc{ ppe-hc4lw_dataset,
    title = { PPE Dataset },
    type = { Open Source Dataset },
    author = { Object Detection PPE },
    howpublished = { \url{ https://universe.roboflow.com/object-detection-ppe-0fljh/ppe-hc4lw } },
    url = { https://universe.roboflow.com/object-detection-ppe-0fljh/ppe-hc4lw },
    journal = { Roboflow Universe },
    publisher = { Roboflow },
    year = { 2022 },
    month = { jun },
    note = { visited on 2022-08-25 },
}
```

## Acknowledgements

<details><summary> <b>Expand</b> </summary>
  
* [https://github.com/WongKinYiu/yolov7]
* [https://github.com/AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)
* [https://github.com/WongKinYiu/yolor](https://github.com/WongKinYiu/yolor)
* [https://github.com/WongKinYiu/PyTorch_YOLOv4](https://github.com/WongKinYiu/PyTorch_YOLOv4)
* [https://github.com/WongKinYiu/ScaledYOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4)
* [https://github.com/Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
* [https://github.com/ultralytics/yolov3](https://github.com/ultralytics/yolov3)
* [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
* [https://github.com/DingXiaoH/RepVGG](https://github.com/DingXiaoH/RepVGG)
* [https://github.com/JUGGHM/OREPA_CVPR2022](https://github.com/JUGGHM/OREPA_CVPR2022)
* [https://github.com/TexasInstruments/edgeai-yolov5/tree/yolo-pose](https://github.com/TexasInstruments/edgeai-yolov5/tree/yolo-pose)

</details>
