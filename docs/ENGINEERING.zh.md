# PDDLVM — 工程概览

[English](ENGINEERING.md) · [中文](ENGINEERING.zh.md)

PDDLVM（物理驱动的深度潜变量模型）通过把参数化 PDE 求解器嵌入分层潜变量模型、
并配合摊还变分推断，来求解 PDE 正问题与反问题。潜参数（传导率、源项、边界条件）
经物理过程生成解场，变分后验从观测中推断这些潜参数。

本文档是包的索引：目录结构、数值层与概率层、已运行的实验（以及所用硬件）、
测试与开发流程。原始论文与代码见 `reference/`。

## 目录结构

```
PDDLVM2/
├── README.md                         快速上手、quickstart、演示
├── CHANGELOG.md                      发布说明（1.1.0 → 2.0.0）
├── LICENSE                           MIT
├── environment.yml                   固定 conda 环境（TF/TFP）
├── pyproject.toml                    包元数据 / 工具配置
├── run.py                            实验运行器（typed JSON 配置）
├── Dockerfile / .dockerignore        容器构建
├── .github/workflows/ci.yml          CI（ruff / mypy / pytest / coverage）
├── configs/
│   └── ve_prior_default.json         默认变分 EM 先验配置
├── assets/                           演示 GIF 与图（文档引用）
│   ├── cylinder_vortex_street.gif    圆柱绕流涡街（伪谱求解器）
│   ├── cylinder_pddlvm_data_vorticity.gif 数据同化：学习场 vs 真值涡量
│   └── cylinder_pddlvm_data_u.png    数据同化：固体区速度抑制
├── pddlvm/
│   ├── __init__.py                   包入口、__version__
│   ├── VEModel.py                    变分 EM 模型（公共 API）
│   ├── NeuralNet.py                  网络：NeuralN、ConvFieldNet(2D/3D)
│   ├── experiment.py                 实验系统（配置、artifacts）
│   ├── plotting.py                   可选出版物风格绘图
│   ├── utilities.py                  共享工具
│   ├── poisson1DPDEState.py          1D Poisson 状态（委托 numerics）
│   ├── poisson1DSolver.py            1D Poisson 求解器（委托 numerics）
│   ├── numerics/                     数值求解器与残差算子
│   │   ├── __init__.py
│   │   ├── fem.py                    向量化 1D 有限元装配（稠密/带状）
│   │   ├── chebyshev.py              Chebyshev 基/变换 + 谱 Poisson
│   │   ├── fd.py                     稳定 Burgers（θ-IMEX、阻尼 Newton）
│   │   ├── mms.py                    制造解、误差范数、收敛阶
│   │   ├── heat3d.py                 3D 热方程求解器
│   │   ├── ns3d.py                   3D Navier–Stokes 残差（NumPy + TF）
│   │   ├── ns2d_obstacle.py          2D 障碍 NS 残差（NumPy + TF）
│   │   ├── ns3d_spectral.py          3D 伪谱 NS（FFT、RK4）
│   │   ├── ns2d_spectral.py          2D 伪谱 NS + 障碍（精确指数积分）
│   │   ├── turbulence.py             Taylor–Green 涡、能量/拟涡能谱
│   │   └── obstacles.py              圆柱/球 SDF、掩模、惩罚场
│   ├── probabilistic/                变分推断与训练
│   │   ├── __init__.py
│   │   ├── distributions.py          对角高斯、残差似然
│   │   ├── elbo.py                   逐分量 ELBO 记录
│   │   ├── trainer.py                tf.data + 目标无关 Trainer
│   │   └── checkpointing.py          Checkpointer（无缝续跑）
│   └── tests/                        单元测试（85 项）
│       ├── __init__.py
│       ├── test_numerics.py          FEM/Chebyshev/Burgers MMS 收敛
│       ├── test_obstacles.py         掩模/惩罚/SDF 性质
│       ├── test_ns3d.py              3D NS 残差（TF↔NumPy、MMS）
│       ├── test_ns3d_obstacles.py    障碍 NS 检查
│       ├── test_ns2d_obstacle.py     2D 障碍 NS 检查
│       ├── test_probabilistic.py     分布、ELBO 算术
│       ├── test_experiment.py        实验系统
│       ├── test_calibration.py       基于仿真的标定
│       ├── test_golden.py            黄金值回归锁
│       ├── test_heat3d.py            3D 热方程检查
│       ├── test_neuraln.py           网络形状 / 初始化
│       ├── test_side_effects.py      导入副作用守卫
│       └── testBasic.ipynb           原始 notebook 演示
├── apps/                             演示驱动（各自写入 artifacts/）
│   ├── burgers_fd.py                 Burgers 有限差分参考
│   ├── burgers_spectral.py           Burgers 伪谱参考
│   ├── burgers_pddlvm.py             含时 Burgers PDDLVM
│   ├── burgers_pinnpddlvm.py         Burgers PDDLVM（PINN 风格）
│   ├── ns_spectral.py                Navier–Stokes 伪谱参考
│   ├── ns_pddlvm.py                  定常 Navier–Stokes PDDLVM
│   ├── sh_spectral.py                Swift–Hohenberg 伪谱参考
│   ├── sh_pddlvm.py                  Swift–Hohenberg PDDLVM
│   ├── heat3d_pddlvm.py              3D 热方程 PDDLVM
│   ├── ns3d_pddlvm.py                3D Navier–Stokes PDDLVM（TGV / ES）
│   ├── cylinder_spectral.py          圆柱绕流谱参考（Re 100/300）
│   └── cylinder_pddlvm.py            圆柱 PDDLVM（解码器 + 残差 + 数据）
├── tools/                            开发 + 实验编排
│   ├── build_docs.py                 离线 API 文档生成
│   ├── coverage.py                   离线行覆盖率估算
│   ├── e9_batch.py / e9_plot.py / e9_animate.py
│   └── e9_L16.py / e9_pillar.py / e9_sphere.py / e9_render3d.py / e9_streamlines.py
├── docs/
│   ├── ENGINEERING.md                English version
│   ├── ENGINEERING.zh.md             本文档
│   └── api/                          生成的 API 文档（tools/build_docs.py）
└── reference/
    ├── PDDLVMPaper/                  原始论文源码（TeX）
    └── PDDLVM-main/                  原始代码快照
```

## 数值层（`pddlvm.numerics`）

- `fem.py` — 向量化 1D 线性有限元装配（稠密与 O(n) 带状），带 Dirichlet 消去。
- `chebyshev.py` — Chebyshev 基/变换与谱 Poisson 求解器。
- `fd.py` — 稳定 Burgers 求解器：线性化 θ-IMEX 时间推进与阻尼 Newton 定常迭代。
- `mms.py` — 制造解框架（误差范数、收敛阶估计），包含 Ethier–Steinman 3D
  Navier–Stokes 解析解。
- `heat3d.py` — 3D 热方程求解器。
- `ns3d.py` / `ns2d_obstacle.py` — 3D 与 2D 障碍 Navier–Stokes 残差算子
  （NumPy + TensorFlow）。
- `ns3d_spectral.py` / `ns2d_spectral.py` — 伪谱 Navier–Stokes 求解器（FFT、
  2/3 去混叠、无散投影、精确指数积分因子）；2D 版本支持浸入式障碍物、平均流
  钉扎与升阻力探针。
- `turbulence.py` — Taylor–Green 涡初值与能量/拟涡能/壳平均能谱。
- `obstacles.py` — 圆柱/球符号距离场、平滑掩模与惩罚场。

每个求解器都用制造解验证，实测收敛阶与理论一致（FEM O(h²)；Chebyshev 谱收敛到
舍入精度），谱 Navier–Stokes 求解器能量守恒到机器精度。

## 概率层（`pddlvm.probabilistic`）

- `distributions.py` — 对角高斯 log-概率 / 重参数化，以及高斯残差似然。
- `elbo.py` — 逐分量 ELBO 记录（log-likelihood / log-prior / log-q / ELBO /
  NELBO），外加物理高斯似然。
- `trainer.py` — `tf.data` 数据集构建器与目标无关的 `Trainer`（梯度带、分量
  指标、LR 调度、早停、checkpoint、可选 `tf.function`）。
- `checkpointing.py` — 支持无缝续跑的 `Checkpointer`。

## 实验系统

`pddlvm.experiment` + `run.py` + typed JSON 配置（`configs/`）提供可复现的流程：
每次运行把 `metrics.json`、图、checkpoint 与训练历史写入 `artifacts/`，按 run
tag 索引。

## 应用（`apps/`）

| 应用 | 说明 |
|---|---|
| `burgers_pddlvm.py` | 含时 Burgers PDDLVM，与瞬态参考对比 |
| `burgers_pinnpddlvm.py` | Burgers PDDLVM（PINN 风格） |
| `burgers_fd.py` / `burgers_spectral.py` | Burgers 有限差分 / 伪谱参考 |
| `ns_pddlvm.py` / `ns_spectral.py` | 定常 Navier–Stokes PDDLVM / 谱参考 |
| `sh_pddlvm.py` / `sh_spectral.py` | Swift–Hohenberg PDDLVM / 谱参考 |
| `heat3d_pddlvm.py` | 3D 热方程 PDDLVM（GIF + PNG） |
| `ns3d_pddlvm.py` | 3D Navier–Stokes PDDLVM（Taylor–Green / Ethier–Steinman） |
| `cylinder_spectral.py` | 圆柱绕流谱参考（Re = 100 / 300，St/Cd/Cl） |
| `cylinder_pddlvm.py` | 2D 圆柱 PDDLVM：神经解码器 + 物理残差 + 数据同化 |

## 实验与结果

所有实验均可通过 `run.py` / `apps/` 驱动与 `tools/` 批量脚本复现；产物写入
`artifacts/`。

### 硬件

| 标签 | 配置 |
|---|---|
| **本地** | Apple Silicon macOS，CPU（TensorFlow 走 `tensorflow-macos`；Python 3.9，TF 2.19，TFP 0.25） |
| **云端** | Linux 云实例，NVIDIA RTX 3080 Ti (12 GB) + 80 vCPU，TF 2.19.1 + TFP 0.25 |
| **GPU 工作站** | NVIDIA RTX 4090D (24 GB) |

### 求解器验证（制造解）

| 实验 | 设置 | 硬件 | 结果 |
|---|---|---|---|
| FEM MMS | 1D Poisson / Burgers，h 加密 | 本地 | 能量阶 1.0、L² 阶 2.0（与理论一致） |
| Chebyshev 谱 MMS | 1D，p 加密 | 本地 | 谱收敛到舍入精度（7.8e-16） |
| Ethier–Steinman MMS | 3D Navier–Stokes，[−1,1]³ | 本地 | 截断缩放 0.145 → 0.044（O(dt + dx²)） |
| 无粘 TGV 能量 | 3D 伪谱，RK4 | 本地 | 能量漂移 2.55e-15（机器精度） |
| 粘性 TGV 衰减 | 2D 伪谱 | 本地 | 与解析衰减一致到机器精度 |

### 圆柱绕流

| 实验 | 设置 | 硬件 | 结果 |
|---|---|---|---|
| 圆柱绕流（谱参考） | Re = 100/200/300，256²/512² 网格，周期盒 + 惩罚 | 云端 | 涡街脱落；升阻力（Cd/Cl）与 Strouhal 数测量 |
| 圆柱 PDDLVM（残差驱动） | Re = 100/300，网格 n = 32–256，nt = 16，12k 步 | RTX 4090D | 残差降 442×–18,600×；固体区 \|u\| → 0.034（GT 0.028）；尾流速度亏损 0.80（GT 0.78） |
| 圆柱 PDDLVM（数据同化） | 稀疏 → 全观测，确定性解码器 | RTX 4090D | 15/15 种子重建全场（相对误差 0.025、相关 0.97） |

### 深度潜变量演示

| 演示 | 方程 | 硬件 |
|---|---|---|
| `heat3d_pddlvm.py` | 3D 热方程 | 本地 |
| `ns3d_pddlvm.py` | 3D Navier–Stokes（Taylor–Green / Ethier–Steinman） | GPU |
| `burgers_pddlvm.py` | 含时 Burgers | 本地 |

## 测试

`python -m unittest discover -s pddlvm/tests -v` 运行完整套件（85 项），覆盖制造解
收敛、能量守恒、副作用守卫、重参数化梯度无偏性、基于仿真的标定、黄金值回归锁与
同种子复现。

## 开发

```bash
ruff check .              # 代码风格检查
mypy pddlvm               # 类型检查
pytest                    # 测试
python tools/coverage.py  # 离线行覆盖率估算
python tools/build_docs.py  # 重新生成 docs/api/
```

## 安装

```bash
conda env create -f environment.yml
conda activate pddlvm2
pip install -e .
```

开发工具（pytest、ruff、mypy）通过 dev extra 安装，或按 `README.md` 所述克隆
固定环境。

## Docker

```bash
docker build -t pddlvm:2.0.0 .
docker run --rm -v "$PWD/artifacts:/work/artifacts" \
    pddlvm:2.0.0 python run.py configs/ve_prior_default.json --steps 200
```

## 许可证

MIT — 见 `LICENSE`。
