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



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--video', required=True)
    args = parser.parse_args()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(args.model, sess_options=options, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise ValueError('视频打不开')
    count = 0
    total = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            tensor, transform = preprocess(frame)
            output = session.run(['output'], {input_name: tensor})[0]
            if count == 0:
                print('模型输入：', tensor.shape, tensor.dtype)
                print('模型输出：', output.shape, output.dtype)
            count += 1
    finally:
        cap.release()
    print('处理帧数：', count, '检测数：', total)



if __name__ == '__main__':
    main()
