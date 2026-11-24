# 秀丽隐杆线虫的姿态 world model 与可解释动力学

课程项目用公开的离食 N2 线虫姿态数据比较 GRU、Dreamer 风格的离散 RSSM、Stephens–Bialek 相位动力学、延迟 Markov 与线性基线。虚拟线虫部分使用公开连接数据运行神经—肌肉—身体模型，并检查新近趋化模拟的可复现性。

- [演示视频](https://junfenglyu.github.io/celegans-world-model/) 包含连接组与后部触碰、观测姿态与 eigenworm 相位、基线与相位嵌入 GRU 的配对滚动。三个 MP4 原文件在 [essay/demos](essay/demos)。
- [论文图件与视频复现指南](essay/README.md) 给出隐藏状态投影、配对仿真和五帧预览的生成命令。
- [中文报告](docs/REPORT.md) 包括文献综述、数据、模型、训练过程、结果和证据范围。
- [公开数据目录](docs/DATA.md) 给出行为、结构连接和功能数据的原始入口与实际用法。

运行下述复现命令后，本地 `artifacts/` 目录包括以下输出。

- [真实姿态与生成轨迹](artifacts/figures/posture_rollouts.mp4) 使用同一段测试历史初始化模型。
- [预测误差图](artifacts/figures/forecast_rmse.png)、[训练曲线](artifacts/figures/training_curves.png)、[随机滚动统计](artifacts/figures/rollout_dynamics.png) 和 [相位漂移](artifacts/figures/phase_drift.png)。
- [modWorm 后部轻触](artifacts/figures/modworm_posterior_touch.mp4)、[趋化试验重跑](artifacts/chemotaxis/chemotaxis_paths.png) 和 [BAAIWorm 已发布肌肉数组](artifacts/virtual/baaiworm_muscles.png)。

## 复现

从此项目根目录运行。实测环境为 Apple Silicon macOS、CPython 3.13、CPU。数据下载使用 curl 和 git，趋化仿真需要 C++17 编译器。既有环境和输出可直接查看，也可以按以下命令重新生成。

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python scripts/fetch_data.py
.venv/bin/python -m wormlab.connectome
.venv/bin/python -m wormlab.experiment
.venv/bin/python -m wormlab.rssm
.venv/bin/python -m wormlab.hybrid
.venv/bin/python -m wormlab.virtual
.venv/bin/python -m wormlab.figures
.venv/bin/python scripts/run_chemotaxis.py
```

运行 `--help` 可查看选项。对行为部分快速试跑时，可加 `--epochs 1 --batches 2 --out artifacts/smoke/benchmark`，RSSM 相应指定 `--benchmark artifacts/smoke/benchmark --out artifacts/smoke/rssm`。上述完整命令使用三个 GRU 随机种子和一个 RSSM 种子。modWorm 的身体 ODE 求解时间较长，日志每完成一个条件更新。

| 路径 | 内容 |
|---|---|
| `wormlab/data.py` | 数据解析、时间窗口与姿态几何 |
| `wormlab/models.py` | 独立相位动力学、延迟 Markov、GRU 与基线模型 |
| `wormlab/experiment.py` / `wormlab/rssm.py` | 拟合、验证选点、留出评估和模型权重 |
| `wormlab/virtual.py` / `scripts/run_chemotaxis.py` | 上游仿真调用、对照及参考数据比较 |
| `artifacts/benchmark` / `artifacts/rssm` | 训练日志 CSV、参数、逐虫指标与滚动预测 |
| `artifacts/connectome` / `artifacts/virtual` / `artifacts/chemotaxis` | 连接组导出、模拟原始输出与元数据 |
| `data/raw` / `external` | 带来源记录的公开数据和上游代码 |

公开输入和参考代码保留各自的出处及使用条款。
