# 检测路线：实现与复现

本项目实现 Python 检测路线。代码位于 `detect.py`，使用 ONNX Runtime CPU 推理和 OpenCV PnP，不训练模型。保留模板的模型、标定、视频和瞄准路线文件。

## 安装与运行

建议 Python 3.12–3.14；本次实际环境为 macOS arm64、Python 3.14.5。依赖固定在 `requirements.txt`。以下命令从仓库根目录执行，创建的环境、下载缓存和结果都在仓库内：

```sh
python3 -m venv .venv
.venv/bin/python -m pip install --no-cache-dir -r detection/requirements.txt
cd detection
../.venv/bin/python detect.py --model assets/0526.onnx --video assets/small.mp4 \
  --calibration assets/calibration.yaml --output results/small-new.jsonl
../.venv/bin/python detect.py --model assets/0526.onnx --video assets/large.mp4 \
  --calibration assets/calibration.yaml --output results/large-new.jsonl
```

Windows 对应解释器路径为 `.venv\Scripts\python.exe`。若固定版本不支持本机平台，需要在独立环境中选用支持该平台的依赖版本。

本次交付已经包含 `results/small.jsonl` 和 `results/large.jsonl`。程序拒绝覆盖已有输出，请为重跑结果指定新路径。运行中的进度和最终统计输出到 stderr，JSONL 文件只包含规定的每帧记录。解码失败、帧数不完整或输入与标定尺寸不符会返回非零状态；这时输出可能是不完整文件，不应当作为完整结果使用。

本机已经装好的环境位于仓库上一级的 `../.vision-venv`（仍在用户指定的工作目录内）。无需重新安装即可从 `detection` 执行：

```sh
../../.vision-venv/bin/python detect.py --help
../../.vision-venv/bin/python -m unittest discover -s tests -v
```

## 算法

1. 顺序读取每一帧。居中 letterbox 到 640×640，补边 114，BGR 转 RGB，除以 255 后排为 NCHW，最终传入连续的 float16 张量。使用 CPUExecutionProvider，输入形状和类型在启动时校验。
2. 输出为 `[1,25200,22]`。前八项直接作为四角点，objectness 使用数值稳定的 sigmoid，颜色和图案各自 argmax。置信度只使用 sigmoid(objectness)。保留 blue、red、neutral、purple 全部颜色。
3. 默认置信度阈值 0.5、NMS IoU 阈值 0.5，与作业建议一致。NMS 对四角点的轴对齐包围框按 objectness 降序、跨类别抑制重复，避免同一装甲板不同类别的重复预测。阈值分别可通过 `--confidence`、`--nms-iou` 修改。
4. 保存实际缩放宽高比和整数补边。用 `(point - padding) / scale` 恢复原图坐标；不按图像横纵坐标重排，不截断超出图像范围的角点。原模型顺序始终为左上、左下、右下、右上。缩放有取整时分别使用实际 x/y 比例。
5. 图案 `1` 和 `base_big` 使用宽 0.230 m，其余使用宽 0.135 m，高均为 0.056 m。选择只依赖图案预测，不读取视频文件名、不依赖帧号。
6. 物理点以板中心为原点，+X 向右、+Y 向下、+Z 向板内：`(-w/2,-h/2,0)`、`(-w/2,h/2,0)`、`(w/2,h/2,0)`、`(w/2,-h/2,0)`。用原始相机矩阵和畸变参数调用 `solvePnPGeneric(SOLVEPNP_IPPE)` 取得候选。每个候选尝试 LM 优化，也保留原始候选，避免优化失败丢失可用解。
7. 排除非有限值、退化/非凸四边形以及任一角点位于相机后方的解。按四点重投影 RMS 选择最优候选。额外设置默认 5 px 的质量门槛（`--max-reprojection-error` 可调），用于拒绝与长方形几何不一致的观测；检测本身仍保留。
8. 对四个物理角点分别计算 `X_camera = R @ X_object + t`。输出单位为米，坐标为每帧相机系（右、下、前），不估计世界坐标或跨帧跟踪。重投影误差按 `sqrt(mean(sum((projected-observed)^2, axis=1)))` 计算，单位为原图像素。
9. 每帧写一个 UTF-8 JSON 对象，帧号从 0 开始，时间固定为 `frame_index / 30`。无检测时写空数组；PnP 失败时两个位姿字段均为 null。序列化禁止 NaN/Infinity。

## 验证

`tests/test_detect.py` 包含 7 项测试，覆盖 RGB/FP16/补边、全部颜色和图案映射、旋转后的点序、objectness 排序的跨类 NMS、空/非有限输出、已知畸变和三维变换的合成 PnP、退化角点和相机后方解。

从 `detection` 目录运行（以下使用新建的仓库内环境）：

```sh
../.venv/bin/python -m unittest discover -s tests -v
../.venv/bin/python verify_results.py --jsonl results/small.jsonl \
  --calibration assets/calibration.yaml --expected-frames 360
../.venv/bin/python verify_results.py --jsonl results/large.jsonl \
  --calibration assets/calibration.yaml --expected-frames 360
```

验证器逐行检查字段、点数、有限数、时间和帧号，检查四个三维点两两距离是否符合物理尺寸，并独立将输出的相机三维点投影回原图，重新计算和核对 RMS。`--video ... --preview ...` 可生成四帧预览，`--report ...` 可保存统计 JSON。验证器中的 360 帧只用于检查本题素材，检测程序不硬编码帧数。

本次实际结果：

| 视频 | 帧数 | 检测数 | 空检测帧 | 位姿失败 | 有效位姿 RMS 中位数 | 有效位姿 RMS 最大值 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| small.mp4 | 360 | 395 | 162 | 0 | 0.1181 px | 0.5561 px |
| large.mp4 | 360 | 150 | 260 | 3 | 0.3240 px | 3.3858 px |

两个视频同时在 CPU 上处理，各约 8 秒（不含安装；具体速度随设备变化）。`results/*-report.json` 保存详细类别统计。预览已人工检查第 0、90、180、270 帧：绿色轮廓表示有效位姿，橙色表示位姿失败，角点标号 0–3 与输出顺序一致。

large 的第 123、137、138 帧各有一个 blue/1 检测未通过 5 px 门槛，其最优候选误差分别约 8.776、9.869、6.193 px，因此输出 null。这些检测的高 objectness 不代表其几何点一定可靠。

## 局限与来源

这些视频没有逐帧人工真值，以上验证证明格式和投影几何的一致性，不是精确率、召回率或真实三维误差的测量。低光、遮挡、截断或类别误判仍可能导致漏检、错检或大小模型选择错误。平面双解按当前帧重投影误差选优，未使用时间连续性。NMS 的轴对齐包围框在相邻目标高度重叠时也可能抑制真实目标。

- [作业网站](https://training.herkules.dev/learn/vision-assignment)：类别映射、几何尺寸和输出规范以该作业为准。
- [教师模板](https://github.com/HKUAstar/vision-training-assignment)：本地起点版本 `c9fa5e9b66b61f5edcc23341ac7dac83a840bcc3`。
- [RobotPilots 参考实现](https://github.com/broalantaps/RobotDetectionModel/blob/babaebd6c8b3aeceda0ca924772b2d70d51801b4/OpenvinoInfer.cpp)：参考其输出结构和后处理概念；本实现自行实现预处理、解码、NMS、位姿选择和输出。没有沿用上游的颜色过滤和类别分数 NMS。
- [OpenCV PnP 文档](https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html)：物体到相机变换、IPPE 和位姿优化。