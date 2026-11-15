from __future__ import annotations

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

# 类别索引映射
COLORS = ('blue', 'red', 'neutral', 'purple')
PATTERNS = ('sentry', '1', '2', '3', '4', '5', 'outpost', 'base_small', 'base_big')


def preprocess(frame):
    """
    为适配模型，先帧预处理，同时返回复原原图坐标需要的参数

    输入OpenCV BGR 图像
    输出形状: (1, 3, 640, 640) float16，以及宽高缩放比例,和补边数据。
    """
    # 等比缩放
    height, width = frame.shape[:2]
    ratio = min(640 / width, 640 / height)
    new_w, new_h = round(width * ratio), round(height * ratio)
    # 创建颜色为114点补边色块
    left, top = (640 - new_w) // 2, (640 - new_h) // 2
    canvas = np.full((640, 640, 3), 114, dtype=np.uint8)
    # 拼接
    canvas[top:top + new_h, left:left + new_w] = cv2.resize(frame, (new_w, new_h))
    # 映射255到1
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    normalized = rgb.astype(np.float32) / 255
    # 调整一下顺序
    tensor = normalized.transpose(2, 0, 1)[None]
    return np.ascontiguousarray(tensor, dtype=np.float16), (new_w / width, new_h / height, left, top)


def nms(corners, scores, threshold):
    """
    用NMS删除重复检测，返回保留候选的索引
    用四角点的轴对齐包围框计算交并比（IoU），按目标置信度从高到低保留

    """
    # 取四角点横纵坐标的分别的最小值和最大值，得到包围框
    lo, hi = corners.min(axis=1), corners.max(axis=1)
    # 计算面积
    areas = np.prod(np.maximum(hi - lo, 0), axis=1)
    # 置信度降序排列，分数相同时保持原先的顺序
    order = np.argsort(-scores, kind='stable')
    keep = []
    while len(order):
        # 保留当前分数最高，然后其余一个一个和最高比过去，看重叠程度，高的话删掉
        index = int(order[0])
        keep.append(index)
        rest = order[1:]
        # 交集宽高：右下角取较小值，左上角取较大值，不相交为零
        overlap = np.maximum(np.minimum(hi[index], hi[rest]) - np.maximum(lo[index], lo[rest]), 0)
        intersection = overlap.prod(axis=1)
        # 并集面积
        union = areas[index] + areas[rest] - intersection
        # IoU = 交集面积 / 并集面积
        iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
        # 删除重叠程度超过阈值的候选
        order = rest[iou <= threshold]
    return keep


def decode(output, transform, confidence=0.5, iou=0.5):
    """
    将模型输出转换成检测列表，恢复原图角点坐标并删除重复检测

    每个候选共22项：
    8 个角点坐标，1 个目标分数，4 个颜色分数，9 个图案分数
    返回的每个检测包含颜色、图案、置信度和二维角点。
    """
    # 数据结构错误输出：
    if output.shape != (1, 25200, 22):
        raise ValueError(f'Unexpected model output shape: {output.shape}')
    rows = output[0]
    # 删除包含NaN或无穷大的候选，防止invalid数字进入后续计算
    # 第 8 项为目标 logit；用数值稳定的 sigmoid 转换为 0～1 的置信度
    # 该写法避免 logit 很负时直接计算 exp(-logit) 发生溢出
    rows = rows[np.isfinite(rows).all(axis=1)]
    logits = rows[:, 8].astype(np.float64)
    scores = np.exp(-np.logaddexp(0, -logits))
    # 只保留达到置信度阈值的候选
    selected = scores >= confidence
    rows, scores = rows[selected], scores[selected]
    # 还原，先减去补边，再除以缩放比例，将角点恢复到原始视频分辨率
    points = rows[:, :8].reshape(-1, 4, 2).astype(np.float64)
    sx, sy, left, top = transform
    points = (points - [left, top]) / [sx, sy]
    detections = []
    for index in nms(points, scores, iou):
        # 颜色和图案分别取最高分的类别；置信度使用sigmoid
        # 保留全部颜色，并转成 Python 标量和列表，方便后续写入JSON
        detections.append(dict(color=COLORS[int(rows[index, 9:13].argmax())],
                               pattern=PATTERNS[int(rows[index, 13:22].argmax())],
                               confidence=float(scores[index]), corners_px=points[index].tolist()))
    return detections


def object_points(pattern):
    """按图案建立装甲板自身坐标系中的四个物理角点，单位为米。"""
    width = 0.230 if pattern in ('1', 'base_big') else 0.135
    # 原点在板中心，高度固定为 0.056 米，因此半高为 0.028 米
    # 点序为左上、左下、右下、右上；四点共面，自身坐标的 Z 均为零
    return np.array([[-width/2, -0.028, 0], [-width/2, 0.028, 0],
                     [width/2, 0.028, 0], [width/2, -0.028, 0]], dtype=np.float64)


def estimate_pose(corners, pattern, matrix, distortion, max_error=5.0):
    """
    用四个二维角点和已知尺寸估计相机坐标系中的三维角点

    输入角点必须是原图像素坐标，matrix 和 distortion 分别为相机内参和畸变参数
    比较平面PnP候选及其优化结果，选择重投影误差最小的有效解
    返回三维角点和四点重投影RMS
    """
    # 转成连续的双精度数组，检查是否恰好有四个有效二维点
    points = np.ascontiguousarray(corners, dtype=np.float64)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        return None, None
    # 装甲板投影应为凸四边形,拒绝交叉、凹陷或面积接近零的角点组合
    if not cv2.isContourConvex(points.astype(np.float32)) or abs(cv2.contourArea(points.astype(np.float32))) < 1e-6:
        return None, None
    model = object_points(pattern)
    # 使用适合共面点的 IPPE，获取可能的旋转向量和平移向量候选
    try:
        solutions = cv2.solvePnPGeneric(model, points, matrix, distortion, flags=cv2.SOLVEPNP_IPPE)
    except cv2.error:
        return None, None
    candidates = []
    # 返回值第一项表示解的数量，没有解时直接报告位姿失败
    if not solutions[0]:
        return None, None
    for rvec, tvec in zip(solutions[1], solutions[2]):
        # 保留原始 IPPE 解，避免后续优化失败或变差时丢失可用结果
        poses = [(rvec, tvec)]
        # 用 LM 优化进一步减小投影误差；传入副本，避免改动原始候选
        try:
            poses.append(cv2.solvePnPRefineLM(model, points, matrix, distortion, rvec.copy(), tvec.copy()))
        except cv2.error:
            pass
        # 分别检查原始位姿和优化后的位姿
        for rotation, translation in poses:
            if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
                continue
            # 旋转向量经 Rodrigues 转成旋转矩阵，再对每个物理点应用 R 和 t
            # 点按行存储，所以写成 model @ R.T，等价于对列向量计算 R @ 点
            rot, _ = cv2.Rodrigues(rotation)
            camera = model @ rot.T + translation.reshape(1, 3)
            # 相机坐标系 +Z 向前，四个角点都必须位于相机前方
            if not np.isfinite(camera).all() or np.any(camera[:, 2] <= 0):
                continue
            # 将物理角点按该位姿投影回原图，使用原始内参和畸变参数
            projected, _ = cv2.projectPoints(model, rotation, translation, matrix, distortion)
            # 先计算每个点的二维距离平方，再对四点取均值并开方，得到 RMS
            error = float(np.sqrt(np.mean(np.sum((projected.reshape(4, 2) - points)**2, axis=1))))
            # 误差单位为原图像素，只接受误差不超过质量门槛的候选
            if math.isfinite(error) and error <= max_error:
                candidates.append((error, camera))
    if not candidates:
        return None, None
    # 在全部有效候选中选择误差最小者，返回四个三维角点及其误差
    error, camera = min(candidates, key=lambda candidate: candidate[0])
    return camera.tolist(), error


def load_calibration(path):
    """读取标定文件，返回相机矩阵、畸变系数和标定图像的宽高。"""
    # 安全解析YAML，再将内参和畸变系数转换成双精度数组
    with Path(path).open(encoding='utf-8') as stream:
        data = yaml.safe_load(stream)
    matrix = np.asarray(data['camera_matrix'], dtype=np.float64)
    distortion = np.asarray(data['distort_coeffs'], dtype=np.float64).reshape(-1)
    # 检查矩阵形状、OpenCV 支持的畸变参数数量、数值有效性及正焦距
    if (matrix.shape != (3, 3) or distortion.size not in (4, 5, 8, 12, 14)
            or not np.isfinite(matrix).all() or not np.isfinite(distortion).all()
            or matrix[0, 0] <= 0 or matrix[1, 1] <= 0):
        raise ValueError('Invalid camera calibration')
    return matrix, distortion, (int(data['image_width']), int(data['image_height']))


def unit_interval(value):
    """供命令行解析器使用：将参数转换成 0～1 范围内的有限浮点数。"""
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise argparse.ArgumentTypeError('must be in [0, 1]')
    return value


def positive_float(value):
    """供命令行解析器使用：将参数转换成大于零的有限浮点数。"""
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError('must be positive and finite')
    return value


def main(argv=None):
    """解析参数，初始化模型，顺序处理视频，并逐帧写出 JSONL。"""
    # 注册四个必填路径，以及置信度、NMS 和位姿误差的可调阈值
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'video', 'calibration', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--confidence', type=unit_interval, default=0.5)
    parser.add_argument('--nms-iou', type=unit_interval, default=0.5)
    parser.add_argument('--max-reprojection-error', type=positive_float, default=5.0)
    args = parser.parse_args(argv)
    # 确认输入文件存在，并拒绝覆盖已有输出文件
    for path in (args.model, args.video, args.calibration):
        if not path.is_file():
            parser.error(f'File not found: {path}')
    if args.output.exists():
        parser.error(f'Output already exists (choose a new path): {args.output}')
    # 读取标定，并创建只使用 CPU 的模型会话，设置内部计算线程数为 4
    matrix, distortion, size = load_calibration(args.calibration)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(args.model), sess_options=options, providers=['CPUExecutionProvider'])
    model_input = session.get_inputs()[0]
    # 检查实际模型接口，防止误用不同输入形状或数据类型的模型
    if model_input.shape != [1, 3, 640, 640] or model_input.type != 'tensor(float16)':
        raise ValueError(f'Unexpected model input: {model_input.shape} {model_input.type}')
    # 打开视频并读取总帧数和帧率；本题时间戳按固定的 30 FPS 计算
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise ValueError(f'Cannot decode video: {args.video}')
    expected_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if abs(fps - 30) > 0.01:
        cap.release()
        raise ValueError(f'This assignment expects 30 FPS, got {fps}')
    # 创建输出目录，初始化已处理帧数、检测总数、位姿失败数和计时器
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = total = failures = 0
    started = time.perf_counter()
    try:
        # 使用独占创建模式打开文件，避免覆盖已有成果；每次循环写一帧
        with args.output.open('x', encoding='utf-8') as stream:
            while True:
                # 顺序读取下一帧，读不到时退出，之后再检查是否读完全部帧
                ok, frame = cap.read()
                if not ok:
                    break
                # 原图尺寸必须与标定一致，否则不能直接使用这组相机内参
                if (frame.shape[1], frame.shape[0]) != size:
                    raise ValueError('Video dimensions do not match calibration')
                # 帧预处理：等比缩放、补边、RGB、归一化和维度转换
                tensor, transform = preprocess(frame)
                # 调用模型，[0] 取出输出列表中的第一个张量
                output = session.run(['output'], {model_input.name: tensor})[0]
                # 解码候选，恢复原图坐标，并通过 NMS 删除重复检测
                detections = decode(output, transform, args.confidence, args.nms_iou)
                # 对每个检测估计位姿；失败时仍保留检测，将位姿相关字段置空
                for detection in detections:
                    camera, error = estimate_pose(detection['corners_px'], detection['pattern'], matrix, distortion,
                                                  args.max_reprojection_error)
                    detection.update(corners_camera_m=camera, reprojection_error_px=error)
                    # boolean可按 0 或 1 累加，统计位姿失败的检测数量
                    failures += camera is None
                # 每帧写一行 JSON，帧号从零开始；视频时间不是程序运行耗时
                # 禁止写出 NaN 或无穷大；Python 的 None 会被序列化成 null
                stream.write(json.dumps(dict(frame_index=count, timestamp_s=count / 30,
                                             detections=detections), allow_nan=False) + '\n')
                # 累计帧数和检测数，每处理 60 帧向标准错误流报告一次进度
                count += 1
                total += len(detections)
                if count % 60 == 0:
                    print(f'{count} frames, {total} detections', file=sys.stderr, flush=True)
    finally:
        # 无论正常结束还是中途报错，都释放视频解码资源
        cap.release()
    # 检查是否完整读取视频，防止将空文件或部分结果误认为处理成功
    if count == 0 or (expected_frames > 0 and count != expected_frames):
        raise ValueError(f'Incomplete decode: {count}/{expected_frames} frames; output is partial')
    # 将最终统计输出到标准错误流，与文件中的逐帧 JSON 数据分开
    print(json.dumps(dict(frames=count, detections=total, pose_failures=failures,
                          elapsed_s=round(time.perf_counter() - started, 2))), file=sys.stderr)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, cv2.error, ort.capi.onnxruntime_pybind11_state.Fail) as error:
        print(f'Error: {error}', file=sys.stderr)
        sys.exit(1)
