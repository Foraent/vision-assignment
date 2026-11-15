"""Validate JSONL independently of inference; optionally render a contact sheet."""
import argparse
from collections import Counter
import json
from pathlib import Path

import cv2
import numpy as np

from detect import COLORS, PATTERNS, load_calibration, object_points


def verify(path, matrix, distortion, expected_frames):
    records = []
    colors, patterns = Counter(), Counter()
    errors, depths = [], []
    failures = 0
    with Path(path).open(encoding='utf-8') as stream:
        for index, line in enumerate(stream):
            record = json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
            assert set(record) == {'frame_index', 'timestamp_s', 'detections'}
            assert record['frame_index'] == index
            assert abs(record['timestamp_s'] - index/30) < 1e-12
            for item in record['detections']:
                assert set(item) == {'color', 'pattern', 'confidence', 'corners_px', 'corners_camera_m', 'reprojection_error_px'}
                assert item['color'] in COLORS and item['pattern'] in PATTERNS
                assert 0 <= item['confidence'] <= 1
                pixels = np.asarray(item['corners_px'])
                assert pixels.shape == (4, 2) and np.isfinite(pixels).all()
                colors[item['color']] += 1
                patterns[item['pattern']] += 1
                if item['corners_camera_m'] is None:
                    assert item['reprojection_error_px'] is None
                    failures += 1
                    continue
                camera = np.asarray(item['corners_camera_m'])
                assert camera.shape == (4, 3) and np.isfinite(camera).all() and (camera[:, 2] > 0).all()
                model = object_points(item['pattern'])
                # Rigid transforms must preserve every pairwise physical distance.
                np.testing.assert_allclose(np.linalg.norm(camera[:, None]-camera[None, :], axis=2),
                                           np.linalg.norm(model[:, None]-model[None, :], axis=2), atol=1e-9)
                projected, _ = cv2.projectPoints(camera, np.zeros(3), np.zeros(3), matrix, distortion)
                rms = float(np.sqrt(np.mean(np.sum((projected.reshape(4, 2)-pixels)**2, axis=1))))
                assert abs(rms - item['reprojection_error_px']) < 1e-6
                errors.append(rms)
                depths.append(float(camera[:, 2].mean()))
            records.append(record)
    assert len(records) == expected_frames, (len(records), expected_frames)
    report = dict(frames=len(records), detections=sum(colors.values()),
                  empty_frames=sum(not r['detections'] for r in records), colors=dict(colors), patterns=dict(patterns),
                  pose_failures=failures, reprojection_rms_median_px=float(np.median(errors)) if errors else None,
                  reprojection_rms_max_px=max(errors, default=None),
                  depth_range_m=[min(depths), max(depths)] if depths else None)
    return records, report


def preview(video, records, output):
    cap = cv2.VideoCapture(str(video))
    tiles = []
    for index in (0, len(records)//4, len(records)//2, 3*len(records)//4):
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        assert ok
        for detection in records[index]['detections']:
            points = np.rint(detection['corners_px']).astype(np.int32)
            color = (0, 255, 0) if detection['corners_camera_m'] is not None else (0, 180, 255)
            cv2.polylines(frame, [points], True, color, 2)
            for n, point in enumerate(points):
                cv2.circle(frame, tuple(point), 4, color, -1)
                cv2.putText(frame, str(n), tuple(point), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
            label = f"{detection['color']} {detection['pattern']} {detection['confidence']:.2f}"
            cv2.putText(frame, label, tuple(points[0]-[0, 12]), cv2.FONT_HERSHEY_SIMPLEX, .6, color, 2)
        frame = cv2.resize(frame, (720, 540))
        cv2.rectangle(frame, (0, 0), (720, 32), (20, 20, 20), -1)
        cv2.putText(frame, f'{Path(video).name} | frame {index} | t={index/30:.2f}s', (10, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1)
        tiles.append(frame)
    cap.release()
    assert cv2.imwrite(str(output), np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jsonl', type=Path, required=True)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--expected-frames', type=int, default=360)
    parser.add_argument('--video', type=Path)
    parser.add_argument('--preview', type=Path)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    matrix, distortion, _ = load_calibration(args.calibration)
    records, report = verify(args.jsonl, matrix, distortion, args.expected_frames)
    if args.preview:
        if not args.video:
            parser.error('--preview requires --video')
        preview(args.video, records, args.preview)
    text = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
    if args.report:
        args.report.write_text(text + '\n', encoding='utf-8')
    print(text)
