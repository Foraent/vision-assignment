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
    height, width = frame.shape[:2]
    ratio = min(640 / width, 640 / height)
    new_w, new_h = (round(width * ratio), round(height * ratio))
    left, top = ((640 - new_w) // 2, (640 - new_h) // 2)
    canvas = np.full((640, 640, 3), 114, dtype=np.uint8)
    canvas[top:top + new_h, left:left + new_w] = cv2.resize(frame, (new_w, new_h))
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    normalized = rgb.astype(np.float32) / 255
    tensor = normalized.transpose(2, 0, 1)[None]
    return (np.ascontiguousarray(tensor, dtype=np.float16), (new_w / width, new_h / height, left, top))


def nms(corners, scores, threshold):
    lo, hi = (corners.min(axis=1), corners.max(axis=1))
    areas = np.prod(np.maximum(hi - lo, 0), axis=1)
    order = np.argsort(-scores, kind='stable')
    keep = []
    while len(order):
        index = int(order[0])
        keep.append(index)
        rest = order[1:]
        overlap = np.maximum(np.minimum(hi[index], hi[rest]) - np.maximum(lo[index], lo[rest]), 0)
        intersection = overlap.prod(axis=1)
        union = areas[index] + areas[rest] - intersection
        iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
        order = rest[iou <= threshold]
    return keep


def decode(output, transform, confidence=0.5, iou=0.5):
    if output.shape != (1, 25200, 22):
        raise ValueError(f'Unexpected model output shape: {output.shape}')
    rows = output[0]
    rows = rows[np.isfinite(rows).all(axis=1)]
    logits = rows[:, 8].astype(np.float64)
    scores = np.exp(-np.logaddexp(0, -logits))
    selected = scores >= confidence
    rows, scores = (rows[selected], scores[selected])
    points = rows[:, :8].reshape(-1, 4, 2).astype(np.float64)
    sx, sy, left, top = transform
    points = (points - [left, top]) / [sx, sy]
    detections = []
    for index in nms(points, scores, iou):
        detections.append(dict(color=COLORS[int(rows[index, 9:13].argmax())], pattern=PATTERNS[int(rows[index, 13:22].argmax())], confidence=float(scores[index]), corners_px=points[index].tolist()))
    return detections


def object_points(pattern):
    width = 0.23 if pattern in ('1', 'base_big') else 0.135
    return np.array([[-width / 2, -0.028, 0], [-width / 2, 0.028, 0], [width / 2, 0.028, 0], [width / 2, -0.028, 0]], dtype=np.float64)


def estimate_pose(corners, pattern, matrix, distortion, max_error=5.0):
    points = np.ascontiguousarray(corners, dtype=np.float64)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        return (None, None)
    if not cv2.isContourConvex(points.astype(np.float32)) or abs(cv2.contourArea(points.astype(np.float32))) < 1e-06:
        return (None, None)
    model = object_points(pattern)
    try:
        solutions = cv2.solvePnPGeneric(model, points, matrix, distortion, flags=cv2.SOLVEPNP_IPPE)
    except cv2.error:
        return (None, None)
    candidates = []
    if not solutions[0]:
        return (None, None)
    for rvec, tvec in zip(solutions[1], solutions[2]):
        poses = [(rvec, tvec)]
        try:
            poses.append(cv2.solvePnPRefineLM(model, points, matrix, distortion, rvec.copy(), tvec.copy()))
        except cv2.error:
            pass
        for rotation, translation in poses:
            if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
                continue
            rot, _ = cv2.Rodrigues(rotation)
            camera = model @ rot.T + translation.reshape(1, 3)
            if not np.isfinite(camera).all() or np.any(camera[:, 2] <= 0):
                continue
            projected, _ = cv2.projectPoints(model, rotation, translation, matrix, distortion)
            error = float(np.sqrt(np.mean(np.sum((projected.reshape(4, 2) - points) ** 2, axis=1))))
            if math.isfinite(error) and error <= max_error:
                candidates.append((error, camera))
    if not candidates:
        return (None, None)
    error, camera = min(candidates, key=lambda candidate: candidate[0])
    return (camera.tolist(), error)


def load_calibration(path):
    with Path(path).open(encoding='utf-8') as stream:
        data = yaml.safe_load(stream)
    matrix = np.asarray(data['camera_matrix'], dtype=np.float64)
    distortion = np.asarray(data['distort_coeffs'], dtype=np.float64).reshape(-1)
    if matrix.shape != (3, 3) or distortion.size not in (4, 5, 8, 12, 14) or (not np.isfinite(matrix).all()) or (not np.isfinite(distortion).all()) or (matrix[0, 0] <= 0) or (matrix[1, 1] <= 0):
        raise ValueError('Invalid camera calibration')
    return (matrix, distortion, (int(data['image_width']), int(data['image_height'])))


def unit_interval(value):
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise argparse.ArgumentTypeError('must be in [0, 1]')
    return value


def positive_float(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError('must be positive and finite')
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'video', 'calibration', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--confidence', type=unit_interval, default=0.5)
    parser.add_argument('--nms-iou', type=unit_interval, default=0.5)
    parser.add_argument('--max-reprojection-error', type=positive_float, default=5.0)
    args = parser.parse_args(argv)
    for path in (args.model, args.video, args.calibration):
        if not path.is_file():
            parser.error(f'File not found: {path}')
    if args.output.exists():
        parser.error(f'Output already exists (choose a new path): {args.output}')
    matrix, distortion, size = load_calibration(args.calibration)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(args.model), sess_options=options, providers=['CPUExecutionProvider'])
    model_input = session.get_inputs()[0]
    if model_input.shape != [1, 3, 640, 640] or model_input.type != 'tensor(float16)':
        raise ValueError(f'Unexpected model input: {model_input.shape} {model_input.type}')
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise ValueError(f'Cannot decode video: {args.video}')
    expected_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if abs(fps - 30) > 0.01:
        cap.release()
        raise ValueError(f'This assignment expects 30 FPS, got {fps}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = total = failures = 0
    started = time.perf_counter()
    try:
        with args.output.open('x', encoding='utf-8') as stream:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if (frame.shape[1], frame.shape[0]) != size:
                    raise ValueError('Video dimensions do not match calibration')
                tensor, transform = preprocess(frame)
                output = session.run(['output'], {model_input.name: tensor})[0]
                detections = decode(output, transform, args.confidence, args.nms_iou)
                for detection in detections:
                    camera, error = estimate_pose(detection['corners_px'], detection['pattern'], matrix, distortion, args.max_reprojection_error)
                    detection.update(corners_camera_m=camera, reprojection_error_px=error)
                    failures += camera is None
                stream.write(json.dumps(dict(frame_index=count, timestamp_s=count / 30, detections=detections), allow_nan=False) + '\n')
                count += 1
                total += len(detections)
                if count % 60 == 0:
                    print(f'{count} frames, {total} detections', file=sys.stderr, flush=True)
    finally:
        cap.release()
    if count == 0 or (expected_frames > 0 and count != expected_frames):
        raise ValueError(f'Incomplete decode: {count}/{expected_frames} frames; output is partial')
    print(json.dumps(dict(frames=count, detections=total, pose_failures=failures, elapsed_s=round(time.perf_counter() - started, 2))), file=sys.stderr)


if __name__ == '__main__':
    main()
