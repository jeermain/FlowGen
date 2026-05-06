# Object Flow Pipeline

这个目录把整条流程拆成三个阶段：

1. `TAPIP3D` 生成稠密 3D 轨迹，输出到 `track_result.npz`。
2. `Grounded-SAM2` 为目标物体生成 `masks=(T,H,W)` 的时序掩码。
3. 一个轻量级后处理模块把全部 3D 轨迹过滤成仅属于目标物体的 flow。

整体实现参考了 `NovaFlow` 的 pipeline 边界设计，但每个阶段都可以独立运行，便于调试和逐步验证。

## 文件说明

- `objflow_schema.py`: 公共数据加载、字段校验和投影辅助函数。
- `generate_grounded_sam2_masks.py`: 基于 TAPIP3D 的结果 NPZ 运行 Grounded-SAM2，生成掩码。
- `filter_object_tracks.py`: 把全量轨迹过滤为目标物体轨迹。
- `visualize_object_tracks.py`: 生成“全量轨迹 vs 目标轨迹”的左右对比视频。
- `run_objflow_pipeline.py`: 顶层多 conda 环境调度器。
- `requirements-postprocess.txt`: 后处理环境的最小依赖列表。

## 输入格式

### Track NPZ

轨迹 NPZ 需要包含以下字段：

- `video`: `(T, C, H, W)`，取值范围在 `[0, 1]` 的浮点数组
- `depths`: `(T, H, W)`
- `intrinsics`: `(T, 3, 3)`
- `extrinsics`: `(T, 4, 4)`，world-to-camera 外参矩阵
- `coords`: `(T, N, 3)`，世界坐标系下的 3D 跟踪点
- `visibs`: `(T, N)`，每条轨迹在每帧的可见性
- `query_points`: 可选，`(N, 4)`，来自 TAPIP3D 的查询点元数据

这与 `TAPIP3D/inference.py` 当前输出的结果一致。

### Mask NPZ

掩码 NPZ 需要包含：

- `masks`: `(T, H, W)`，类型为 `uint8` 或 `bool`

如果分辨率不一致，过滤脚本会先把 mask 重采样到轨迹视频的分辨率；但帧数必须一开始就严格一致。

## 过滤逻辑

对象级过滤分两步：

1. `frame0 membership`: 先把 TAPIP3D 的查询点投影到第 0 帧，只保留落在目标 mask 内的点。
2. `temporal consistency`: 再把保留下来的 3D 轨迹逐帧投影回图像平面，只保留在足够多可见帧中持续落在传播后 mask 内的轨迹。

最终输出文件为 `object_tracks.npz`，其中包含：

- `coords_object`
- `visibs_object`
- `query_points_object`
- `object_indices`
- `masks`
- `temporal_ratio`
- `visible_counts`
- `inside_counts`

## 推荐环境

开发阶段建议把各个阶段分开管理：

### `tapip3d`

直接使用 `TAPIP3D/README.md` 里现有的环境配置。

### `grounded-sam2`

依赖建议从以下位置安装：

- `NovaFlow/server/grounded_sam_2/grounding_dino/requirements.txt`
- `NovaFlow/server/grounded_sam_2` 目录下的 SAM2 本体

### `objflow-post`

```bash
conda create -n objflow-post python=3.10
conda activate objflow-post
pip install -r /mnt/nas/yangrun/object_flow_pipeline/requirements-postprocess.txt
```

## 从已有 TAPIP3D 结果开始运行

```bash
python /mnt/nas/yangrun/object_flow_pipeline/run_objflow_pipeline.py \
  --track-npz /path/to/track.result.npz \
  --prompt "mug" \
  --output-dir /path/to/objflow_run
```

运行后会产出：

- `segmentation_masks.npz`
- `segmentation_overlay.mp4`
- `object_tracks.npz`
- `object_tracks_compare.mp4`

## 运行完整三阶段流程

```bash
python /mnt/nas/yangrun/object_flow_pipeline/run_objflow_pipeline.py \
  --input-path /path/to/video.mp4 \
  --prompt "mug" \
  --output-dir /path/to/objflow_run
```

## 分阶段单独运行

### 1. 只生成 masks

```bash
python /mnt/nas/yangrun/object_flow_pipeline/generate_grounded_sam2_masks.py \
  --track-npz /path/to/track.result.npz \
  --prompt "mug" \
  --output-mask-npz /path/to/segmentation_masks.npz \
  --output-video /path/to/segmentation_overlay.mp4
```

### 2. 只过滤轨迹

```bash
python /mnt/nas/yangrun/object_flow_pipeline/filter_object_tracks.py \
  --track-npz /path/to/track.result.npz \
  --mask-npz /path/to/segmentation_masks.npz \
  --output-npz /path/to/object_tracks.npz
```

### 3. 只可视化对象轨迹

```bash
python /mnt/nas/yangrun/object_flow_pipeline/visualize_object_tracks.py \
  --track-npz /path/to/track.result.npz \
  --object-npz /path/to/object_tracks.npz \
  --output-video /path/to/object_tracks_compare.mp4
```

## 它和 NovaFlow 的区别

`NovaFlow` 会把多个项目 vendored 到 `server/` 目录下，然后通过单一 Python 环境配合 `sys.path` 和 `PYTHONPATH` 完成统一加载。这种方式很适合固定部署，但在你还处于多上游项目混合迭代阶段时，调试成本更高。

这个目录刻意保留了和 `NovaFlow` 相同的逻辑阶段边界，但阶段之间通过独立 conda 环境和文件交接。这样做的好处是：

- 每个阶段都可以独立调试
- 更容易规避 PyTorch/CUDA 版本冲突
- 某一阶段失败后，只需要重跑该阶段

## 什么时候再收敛成单环境

建议先维持多环境方案，直到以下条件都稳定下来：

- PyTorch/CUDA 的准确版本已经固定
- SAM2 和 TAPIP3D 的依赖不再冲突
- 你确实需要一个单命令或单服务的稳定生产入口

到那时，就可以逐步向 `NovaFlow` 的模式靠拢：

1. 把需要的仓库统一整理到一个运行时目录
2. 用一个 Docker 镜像统一依赖
3. 把当前基于 `conda run -n ...` 的阶段交接，替换成进程内导入或 Ray worker 调度

在那之前，当前这套多环境 runner 是更稳妥的默认方案。
