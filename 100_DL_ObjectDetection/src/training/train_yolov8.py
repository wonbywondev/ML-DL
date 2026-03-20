#!/usr/bin/env python3
import argparse, os
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument('--data', default='configs/dataset.yaml')
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--batch', type=int, default=16)
args = parser.parse_args()

if not os.path.exists(args.data):
    raise FileNotFoundError(f"{args.data} not found")

model = YOLO('yolov8n.pt')
model.train(data=args.data, epochs=args.epochs, imgsz=640, batch=args.batch,
            iou=0.6, max_det=100, project='runs/detect', name='pill_baseline_v1')
