## This repository contains:

- trained .pt and .pth models (YOLOv8 & Detectron2)
- prediction scripts
- test-images
- evaluation script with metrics


---


## Repository Structure
```text
repo/
│
├─ README.md
├─ LICENSE                 # MIT (for evaluation + output)
├─ NOTICE.md               # Third-party attribution
│
├─ models/
│   ├─ yolov8/
│   │   ├─ model.pt
│   │   └─ LICENSE        # AGPL-3.0
│   │
│   └─ detectron2/
│       ├─ model.pth
│       └─ LICENSE        # Apache-2.0
│
├─ prediction/
│   ├─ yolov8/
│   │   ├─ predict_yolo.py
│   │   └─ LICENSE        # AGPL-3.0
│   │
│   └─ detectron2/
│       ├─ predict_detectron2.py
│       └─ LICENSE        # Apache-2.0
│
├─ evaluation/
│   ├─ evaluate.py
│   └─ LICENSE            # MIT
│
├─ test_images/
│   ├─ img1.jpg
│   └─ img2.jpg
│
└─ example_outputs/
    ├─ yolo_txt/
    └─ detectron2_txt/


---

## Licensing

This repository contains components under different licenses:

MIT: evaluation pipeline and output data

AGPL-3.0: YOLOv8 prediction scripts and YOLO model weights

Apache-2.0: Detectron2 prediction scripts and Detectron2 model weights

See the LICENSE files in each folder for details.


---


## Requirements
supplement


---


## Usage Overview
Place your images in test_images/

## 1. YOLOv8 Prediction:
run:
python prediction/yolov8/predict_yolo.py

Output text files are saved in:
example_outputs/yolo_txt/

## 2. Detectron2 Prediction:
run:
python prediction/detectron2/predict_detectron2.py

Output text files are saved in:
example_outputs/detectron2_txt/

## 3. Evaluation:
run:
python evaluation/evaluate.py

This script reads the text files from the prediction
output folders and computes metrics.


---


## Notes

The YOLOv8seg model was fine-tuned using Ultralytics YOLOv8 (AGPL-3.0).
The Detectron2 model was fine-tuned using Detectron2 (Apache-2.0).

