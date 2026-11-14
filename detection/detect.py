import argparse
import json
import math
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import onnxruntime as ort
import yaml

COLORS = ('blue', 'red', 'neutral', 'purple')
PATTERNS = ('sentry', '1', '2', '3', '4', '5', 'outpost', 'base_small', 'base_big')


def preprocess(frame):
    # 先保留比例缩放，再居中补边
    height, width = frame.shape[:2]
    ratio = min(640 / width, 640 / height)
    new_w, new_h = round(width * ratio), round(height * ratio)
    left, top = (640 - new_w) // 2, (640 - new_h) // 2
    canvas = np.full((640, 640, 3), 114, dtype=np.uint8)
    canvas[top:top + new_h, left:left + new_w] = cv2.resize(frame, (new_w, new_h))
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    image = rgb.astype(np.float32) / 255
    tensor = image.transpose(2, 0, 1)[None]
    return np.ascontiguousarray(tensor, dtype=np.float16), (new_w / width, new_h / height, left, top)



def nms(corners, scores, threshold):
    # 先按分数排序，再逐个判断包围框交并比
    lo, hi = corners.min(axis=1), corners.max(axis=1)
    order = sorted(range(len(scores)), key=lambda index: -scores[index])
    keep = []
    while order:
        index = order.pop(0)
        keep.append(index)
        remaining = []
        for other in order:
            overlap = np.maximum(np.minimum(hi[index], hi[other]) - np.maximum(lo[index], lo[other]), 0)
            intersection = float(overlap[0] * overlap[1])
            area_a = float(np.prod(hi[index] - lo[index]))
            area_b = float(np.prod(hi[other] - lo[other]))
            union = area_a + area_b - intersection
            iou = intersection / union if union > 0 else 0
            if iou <= threshold:
                remaining.append(other)
        order = remaining
    return keep



def decode(output, transform, confidence=0.5, iou=0.5):
    sx, sy, left, top = transform
    detections = []
    for row in output[0]:
        score = float(1 / (1 + np.exp(-float(row[8]))))
        if score < confidence:
            continue
        points = row[:8].reshape(4, 2).astype(np.float64)
        points = (points - [left, top]) / [sx, sy]
        detections.append(dict(
            color=COLORS[int(row[9:13].argmax())],
            pattern=PATTERNS[int(row[13:22].argmax())],
            confidence=score,
            corners_px=points.tolist(),
        ))
    if not detections:
        return []
    corners = np.asarray([item['corners_px'] for item in detections])
    scores = np.asarray([item['confidence'] for item in detections])
    return [detections[index] for index in nms(corners, scores, iou)]



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--video', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--confidence', type=float, default=0.5)
    parser.add_argument('--nms-iou', type=float, default=0.5)
    args = parser.parse_args()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(args.model, sess_options=options, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise ValueError('视频打不开')
    stream = open(args.output, 'w', encoding='utf-8')
    count = 0
    total = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            tensor, transform = preprocess(frame)
            output = session.run(['output'], {input_name: tensor})[0]
            detections = decode(output, transform, args.confidence, args.nms_iou)
            stream.write(json.dumps(dict(frame_index=count, timestamp_s=count / 30, detections=detections)) + '\n')
            total += len(detections)
            count += 1
    finally:
        cap.release()
        stream.close()
    print('处理帧数：', count, '检测数：', total)



if __name__ == '__main__':
    main()
