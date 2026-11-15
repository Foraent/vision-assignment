import unittest
from unittest.mock import patch

import cv2
import numpy as np

from detect import COLORS, PATTERNS, decode, estimate_pose, load_calibration, object_points, preprocess


class DetectionTests(unittest.TestCase):
    def setUp(self):
        self.matrix = np.array([[1200., 0, 720], [0, 1200, 540], [0, 0, 1]])
        self.distortion = np.array([-.08, .13, -.0004, -.0008, 0.])

    def test_rgb_fp16_letterbox(self):
        frame = np.full((1080, 1440, 3), [10, 20, 240], dtype=np.uint8)
        tensor, transform = preprocess(frame)
        self.assertEqual(tensor.shape, (1, 3, 640, 640))
        self.assertEqual(tensor.dtype, np.float16)
        self.assertTrue(tensor.flags.c_contiguous)
        np.testing.assert_allclose(tensor[0, :, 100, 100], np.array([240, 20, 10]) / 255, atol=.001)
        np.testing.assert_allclose(tensor[0, :, 0, 0], 114/255, atol=.001)
        self.assertEqual(transform, (640/1440, 480/1080, 0, 80))

    def test_decode_all_classes_no_reordering(self):
        output = np.full((1, 25200, 22), -100, np.float32)
        original = np.array([[50, 60], [30, 80], [60, 110], [80, 90]])
        transform = (.4, .4, 0, 100)
        for i in range(9):
            points = original + [i * 200, 0]
            output[0, i, :8] = (points * .4 + [0, 100]).ravel()
            output[0, i, 8] = np.log(4)
            output[0, i, 9 + i % 4] = 10
            output[0, i, 13 + i] = 10
        result = decode(output, transform)
        self.assertEqual(len(result), 9)
        for i, item in enumerate(result):
            self.assertEqual(item['color'], COLORS[i % 4])
            self.assertEqual(item['pattern'], PATTERNS[i])
            self.assertAlmostEqual(item['confidence'], .8, places=6)
            np.testing.assert_allclose(item['corners_px'], original + [i * 200, 0], atol=1e-4)

    def test_nms_ranks_objectness_across_classes(self):
        output = np.full((1, 25200, 22), -100, np.float32)
        for i, logit in enumerate([2, 3]):
            output[0, i, :8] = [10, 10, 10, 30, 50, 30, 50, 10]
            output[0, i, 8] = logit
            output[0, i, 9+i] = 1
            output[0, i, 13+i] = 100-i*99
        result = decode(output, (1, 1, 0, 0))
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['pattern'], '1')
        self.assertAlmostEqual(result[0]['confidence'], 1/(1+np.exp(-3)))

    def test_empty_and_nonfinite(self):
        output = np.full((1, 25200, 22), -100, np.float32)
        output[0, 0, 8] = np.nan
        self.assertEqual(decode(output, (1, 1, 0, 0)), [])

    def test_pose_rotation_distortion_and_geometry(self):
        for pattern in PATTERNS:
            with self.subTest(pattern=pattern):
                model = object_points(pattern)
                rvec = np.array([.25, -.4, 2.7])
                tvec = np.array([.1, -.2, 3.])
                observed, _ = cv2.projectPoints(model, rvec, tvec, self.matrix, self.distortion)
                camera, error = estimate_pose(observed.reshape(4, 2), pattern, self.matrix, self.distortion)
                expected = model @ cv2.Rodrigues(rvec)[0].T + tvec
                np.testing.assert_allclose(camera, expected, atol=1e-6)
                self.assertLess(error, 1e-6)

    def test_bad_pose_returns_nulls(self):
        for points in (np.zeros((4, 2)), np.full((4, 2), np.nan), [[0, 0], [0, 2], [2, 0], [2, 2]]):
            self.assertEqual(estimate_pose(points, '3', self.matrix, self.distortion), (None, None))

    def test_rejects_behind_camera(self):
        points = np.array([[10., 10], [10, 20], [40, 20], [40, 10]])
        with patch('detect.cv2.solvePnPGeneric', return_value=(1, [np.zeros(3)], [np.array([0., 0, -3])])), \
             patch('detect.cv2.solvePnPRefineLM', side_effect=cv2.error('failed')):
            self.assertEqual(estimate_pose(points, '3', self.matrix, self.distortion), (None, None))


if __name__ == '__main__':
    unittest.main()
