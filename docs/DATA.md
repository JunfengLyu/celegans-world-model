# 公开数据与本地使用

| 数据 | 测量对象与访问 | 本项目用途 |
|---|---|---|
| Broekmans et al. 2016 / Ahamed et al. 2021 | 12 条离食 N2，35 分钟/虫。Dryad 的原始成像为 32 Hz，追踪后为 16 Hz。作者代码仓库提供 `crawl.mat` 五个姿态系数和 `EigenWorms.mat` 切向角基底。 [原始数据](https://doi.org/10.5061/dryad.t0m6p) · [数据字典及代码](https://bitbucket.org/tosifahamed/behavioral-state-space/src/ad125e08db4537379371d89e490bf913f7b5e510/) | 本地训练、跨虫验证、姿态动画。|
| Costa et al. 2024 | 论文数据包含离食姿态、轮廓、位移及预计算中间量。Kaggle v2 发布包约 56 GB，页面标注 CC BY-NC-ND 4.0。 [论文](https://doi.org/10.1073/pnas.2318805121) · [数据](https://doi.org/10.34740/kaggle/ds/3882219) · [代码](https://github.com/AntonioCCosta/markov_worm) | 延迟 Markov 方法参考。拟合输入使用上行的 Ahamed 发布文件。 |
| Cook et al. 2019 | 成体雌雄同体与雄虫的 EM 网络。SI5 2020 修订表的边权以 EM 切片计量。化学边有方向，缝隙连接采用发布的对称矩阵。 [论文](https://doi.org/10.1038/s41586-019-1352-7) · [矩阵入口](https://wormwiring.org/pages/adjacency.html) | 本地保留原始 XLSX，导出雌雄同体化学和电连接 CSV、矩阵及统计。modWorm Cook 情景取与上游命名匹配的 279 个神经元。 |
| WormAtlas / modWorm | `NeuronConnect.xls` 提供旧版本连接条目，`NeuronFixedPoints.xls` 提供肌肉映射输入。上游 modWorm 再叠加其连接和映射修正矩阵。 [Varshney 论文](https://doi.org/10.1371/journal.pcbi.1001066) · [连接表](https://www.wormatlas.org/images/NeuronConnect.xls) · [映射表](https://www.wormatlas.org/images/NeuronFixedPoints.xls) · [modWorm 代码](https://github.com/shlizee/modWorm) | 用于运行上游网络和肌肉模块。 |
| Witvliet et al. 2021 | 8 个发育阶段样本的重建，可看保守、可变和随发育改变的连接。 [论文](https://doi.org/10.1038/s41586-021-03778-8) · [源数据与分析](https://github.com/dwitvliet/nature2021) | 数据目录参考。 |
| Randi et al. 2023，Princeton | 对头部单神经元进行光刺激并记录全脑钙活动，报告 23,433 个神经元对的测量。响应可经多跳或突触外信号传递。 [论文](https://doi.org/10.1038/s41586-023-06683-4) · [OSF](https://doi.org/10.17605/OSF.IO/E2SYT) · [交互图谱](https://funconn.princeton.edu/) | 未来可针对刺激响应约束或验证模拟中的参数。 |
| Atanas & Kim et al. 2023 | 自由移动中的脑广域钙信号与行为读数，部分有 NeuroPAL 标识。论文主分析使用 14 条在稀疏食物上自由探索的个体。WormWideWeb 还汇集多个研究的 HDF5 和 JSON。 [论文](https://doi.org/10.1016/j.cell.2023.07.035) · [数据版本和字段](https://wormwideweb.org/about/datasets/) | 将姿态模型扩展到神经活动条件预测的候选数据。 |
| Yemini et al. 2013 / Tierpsy | Worm Tracker 2.0 论文初始版本含 305 个品系、9,203 段视频及行为特征。Tierpsy 为后续追踪工具。 [论文](https://doi.org/10.1038/nmeth.2560) · [实验室入口](https://www.yeminilab.com/behavior) · [Tierpsy](https://github.com/Tierpsy/tierpsy-tracker) | 跨品系、摄食条件和追踪流程的外部分布检验候选。 |
| Zhao et al. BAAIWorm 公开演示数组 | 原仓库随附肌肉目标序列、开环导出和配置。目标为模拟流程的控制数据，具体数组的逐样本对齐未建立。 [论文](https://doi.org/10.1038/s43588-024-00738-w) · [固定版本数组](https://github.com/Jessie940611/BAAIWorm/tree/b99ca5ab9c00aef739fceb1cf007bc50b06f2ccc/eworm/ghost_in_mesh_sim/data/tuned/video_offline) | 本地读入和重绘。 |
| Squires et al. 2026 | 作者仓库含圆锥梯度的角度扫描、配置、模拟输出和遗传算法。这里下载 120° 条件的配置、碰撞对象和参考 `body.dat`。 [预印本](https://doi.org/10.64898/2026.08.09.743732) · [固定版本代码](https://github.com/adasquires/celegans-chemotaxis/tree/5f639a23d546093f1b278fa5d9c96fdc99da5f49) | 使用作者 C++ 重新积分，并与参考输出比较。 |

`crawl.mat` 的 HDF5 根变量 `tr` 是 MATLAB cell 引用数组。每条记录长度 33,600 帧。取每条记录的前 5 列，保持 cell 顺序和原始时间索引。所有未来预测窗口要求历史和未来姿态系数均有限，不插值。个体编号是 cell 的 1 起始顺序。Dryad 说明用晚 L4 表述记录个体，Costa 论文用 young-adult 表述再分析样本，报告保留发布版本并统一称“离食 N2”。[Dryad 记录及说明](https://datadryad.org/dataset/doi%3A10.5061/dryad.t0m6p) · [Costa et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC11317559/)

行为窗口、Cook 结构和 Randi 或 Atanas 的功能数据各有独立的动物来源、记录条件和测量方式。将不同数据按神经元命名配准可以用来约束模型。评估逐虫预测或刺激响应需要相应实验中的配对观测。输入下载清单及 SHA-256 在 [fetch_data.py](../scripts/fetch_data.py)，化学和电连接导出字段在 [connectome.py](../wormlab/connectome.py)。
