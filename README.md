# 视觉组作业A

uid:3036896065

email: u3689606@connect.hku.hk

WechatNO: allen317609

实现装甲板检测，PnP角点估计和JSONL输出。

代码位于[detection/detect.py](detection/detect.py)。
结果位于 [detection/results](detection/results)。

## 算法介绍

1. 顺序读取每一帧。居中 letterbox 到 640×640，补边 114，BGR 转 RGB，除以 255 后排为 NCHW，最终传入连续的 float16 张量。
2. 输出为 `[1,25200,22]`。前八项直接作为四角点，objectness 使用数值稳定的 sigmoid，颜色和图案各自argmax。
3. 默认置信度阈值 0.5、NMS IoU 阈值 0.5，与作业建议一致。NMS 对四角点的轴对齐包围框按 objectness 降序、跨类别抑制重复，避免同一装甲板不同类别的重复预测。
4. 保存实际缩放宽高比和整数补边。用 `(point - padding) / scale` 恢复原图坐标。
5. 图案 `1` 和 `base_big` 使用宽 0.230 m，其余使用宽 0.135 m，高均为 0.056 m。
6. 物理点以板中心为原点，+X 向右、+Y 向下、+Z 向板内：`(-w/2,-h/2,0)`、`(-w/2,h/2,0)`、`(w/2,h/2,0)`、`(w/2,-h/2,0)`。用原始相机矩阵和畸变参数调用 `solvePnPGeneric(SOLVEPNP_IPPE)` 取得候选。
7. 排除非有限值、退化/非凸四边形以及任一角点位于相机后方的解。按四点重投影 RMS 选择最优候选。额外设置默认 5 px 的质量门槛（`--max-reprojection-error` 可调），用于拒绝与长方形几何不一致的观测。
8. 对四个物理角点分别计算 `X_camera = R @ X_object + t`。输出单位为米，坐标为每帧相机系（右、下、前），不估计世界坐标或跨帧跟踪。重投影误差按 `sqrt(mean(sum((projected-observed)^2, axis=1)))` 计算，单位为原图像素。
9. 每帧写一个 UTF-8 JSON 对象，帧号从 0 开始，时间固定为 `frame_index / 30`。

# lyf 的第三次求助

[中文](README.md) · [English](README.en.md)

## 云台能动了，然后呢？

在[控制组的作业](https://github.com/HKUAstar/control-training-template)中，
lyf 把 yaw、pitch 云台教具交给了控制组。这次我们假设他们已经完成程序：
云台可以回正、跟随目标角度，也能在调试时停止输出。

看到云台终于不再反复点头，lyf 松了一口气：“现在能不能让它自己盯着目标转？”

控制组也想试试。他们留出了目标角度接口，准备接上视觉系统。
但视觉组还得回答两个问题：相机画面里的哪一块是装甲板？
就算知道它现在在哪里，等弹丸飞过去，它又会转到哪里？

这一次，你负责实现视觉链路中的一段。为了让你能在自己的电脑上独立完成，
我们提供录像、模型、标定和瞄准仿真器。本次不需要真实教具、相机、电机或控制板，
也不要求与控制组固件通信。故事中的联调只是背景，不是额外的硬件任务。

## 两条路线，任选其一

**只需完成检测或瞄准中的一项，不要求两项都做。两条路线互不依赖。**

| 路线 | 你负责什么 | 已提供什么 | 从哪里开始 |
| --- | --- | --- | --- |
| 检测 | 从视频检测装甲板，用 PnP 恢复三维角点，导出 JSONL | 预训练模型、比赛片段、相机标定 | [detection/README.md](detection/README.md) |
| 瞄准 | 根据延迟送达的目标状态预测装甲板位置，决定瞄准角和开火时机 | 精确的历史观测、弹道工具、3D 仿真、MCAP 回放 | [aiming/README.md](aiming/README.md) |

检测路线语言不限，推荐 Python、C++ 或 Rust，不需要训练模型。
瞄准路线使用 Rust，不需要实现检测器、跟踪器或状态估计。
两者都允许使用参考资料和 AI 工具，但你需要能够解释自己的实现并定位问题。

## 开始与提交

1. 打开[作业模板](https://github.com/HKUAstar/vision-training-assignment)，点击
   **Use this template → Create a new repository**，在自己的账号下创建作业仓库。
2. Clone 自己的仓库，选择检测或瞄准路线，按对应 README 完成作业。
3. 将完成的代码 commit 并 push 到自己的仓库，然后提交该仓库的 **GitHub URL**。
