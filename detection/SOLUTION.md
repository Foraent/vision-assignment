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