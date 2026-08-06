# 农作物病害检测 (Plants Disease Detection)

> AI Challenger 2018 [农作物病害检测](https://challenger.ai/competition/pdr2018) 竞赛 baseline，基于 PyTorch 的图像分类方案。
>
> **声明**：开源仅供交流学习，数据请勿用于商业用途。转载或解读请注明出处，谢谢！

**成绩**：线上 0.8805，线下 0.875（因划分存在随机性，复现可能有波动，已尽量固定随机种子）。

---

## 目录

- [环境依赖](#环境依赖)
- [数据准备](#数据准备)
- [使用方法](#使用方法)
- [方案说明](#方案说明)
- [项目结构](#项目结构)
- [2026 现代化改造记录](#2026-现代化改造记录)
- [相关链接](#相关链接)

---

## 环境依赖

代码已从原始的 `python3.6 / pytorch0.4.1` 迁移到 **PyTorch 2.x**，并支持 **CPU / GPU 自动切换**与**混合精度 (AMP)**。

```bash
pip install -r requirements.txt
```

主要依赖：

| 包 | 版本要求 |
| --- | --- |
| python | >= 3.8 |
| torch | >= 2.0 |
| torchvision | >= 0.15 |
| scikit-learn | 用于分层划分 |
| pandas / numpy | 数据处理 |
| pillow / opencv-python / scikit-image | 图像与离线增强 |
| tqdm | 进度条 |

运行设备在 [`config.py`](config.py) 中自动探测：有 GPU 时用 CUDA + AMP，无 GPU 时回退到 CPU。

## 数据准备

1. 下载数据集（10 月 23 日更新后的新版，含训练/验证/测试 A、B）：
   [百度网盘](https://pan.baidu.com/s/16f1nQchS-zBtzSWn9Guyyg)，提取码：`iksk`
2. 将**测试集**图片复制到 `data/test/` 下。
3. 将**训练集 + 验证集**的图片都复制到 `data/temp/images/`，两个 `json` 标注文件放到 `data/temp/labels/`。
4. 执行 `move.py` 按类别整理图片到 `data/train/<类别>/`：

   ```bash
   python move.py
   ```

   > 该脚本会删除样本异常的第 44、45 类，并把之后的类别编号整体前移 2 位（`num_classes` 因此为 59）。`main.py` 的 `test()` 会做逆映射，把预测结果还原回官方原始编号。

## 使用方法

```bash
# 训练 + 在测试集上推理，生成提交文件
python main.py
```

- 训练日志写入 `logs/log_train.txt`；
- 权重保存在 `checkpoints/`，最优模型在 `checkpoints/best_model/`；
- 训练结束后自动加载最优模型对测试集推理，结果写入 `submit/baseline.json`（官方提交格式）。

关键超参数（在 [`config.py`](config.py) 中调整）：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `epochs` | 40 | 训练轮数 |
| `batch_size` | 8 | 650×650 大图较吃显存，按需调整 |
| `img_height/img_width` | 650 | 输入尺寸 |
| `lr` | 1e-4 | Adam 学习率 |
| `use_amp` | True | 混合精度（仅 CUDA 生效） |
| `use_focal_loss` | False | 置 True 改用（已修复的）FocalLoss |

## 方案说明

- **模型**：ResNet50（ImageNet 预训练），替换全局池化为 `AdaptiveAvgPool2d(1)` 并接 59 类分类头。
- **数据划分**：合并 train/val 后用 `StratifiedKFold` 思路做分层随机划分（`test_size=0.15`）。
- **数据增强**（在线，见 [`dataset/dataloader.py`](dataset/dataloader.py)）：
  `RandomRotation(30)`、`RandomHorizontalFlip`、`RandomVerticalFlip`、`RandomAffine(45)`。
- **离线增强**（可选，见 [`data_aug.py`](data_aug.py)）：高斯噪声、亮度/对比度变化、翻转等。
- **优化器**：Adam（`amsgrad=True`），`StepLR`（每 10 epoch × 0.1）。
- **损失**：默认 `CrossEntropyLoss`，可切换 FocalLoss。

## 项目结构

```
.
├── config.py              # 全部超参数、设备与开关
├── move.py                # 按标注把图片整理进类别目录
├── data_aug.py            # 离线数据增强（可选）
├── dataset/dataloader.py  # Dataset / DataLoader / transforms
├── models/model.py        # ResNet50 分类网络
├── utils.py               # AverageMeter / accuracy / Logger / FocalLoss 等
└── main.py                # 训练、评估、测试主流程
```

## 2026 现代化改造记录

在原 2018 版基础上做了如下修复与升级：

**Bug 修复**
- **FocalLoss 数学错误**：原实现对 batch 平均后的标量做加权，退化成普通交叉熵。现改为 `reduction='none'` 逐样本计算调制因子 `(1-pt)^γ`，并支持 `mean/sum/none`。
- **删除坏掉的 `DenseModel`**：forward 里定义了却未使用的层、写死的 pool 尺寸、`sigmoid` + `CrossEntropyLoss` 双重激活等问题，整体移除（训练本就走 `get_net`）。
- **`data_aug.py` 语法错误**：修复错位的 `try/except` 缩进。

**PyTorch 2.x 迁移**
- 移除已废弃的 `torch.autograd.Variable`，改用 `tensor.to(device)`。
- 预训练加载 `pretrained=True` → `weights=ResNet50_Weights.IMAGENET1K_V1`。
- `scheduler.step(epoch)` 旧用法 → `scheduler.step()`，并移到 epoch 末尾。
- `torch.load(..., weights_only=False, map_location=device)` 显式化，兼容新默认值。
- Pillow 常量 `Image.FLIP_*` → `Image.Transpose.FLIP_*`（新版已移除旧别名）。

**新增能力**
- **CPU / GPU 自动切换**：无显卡也能跑。
- **混合精度 (AMP)**：`use_amp` 开关，CUDA 上显著省显存、提速。
- `DataLoader` 增加 `num_workers`，读图统一 `convert("RGB")` 防止灰度/RGBA 崩溃。
- 修正 `img_weight` → `img_width` 命名及 `Resize` 的 (H, W) 顺序。

## 相关链接

- 完整代码：[plants_disease_detection](https://github.com/spytensor/plants_disease_detection)
- 图像分类入门教程与代码：[从实例掌握 pytorch 进行图像分类](http://www.spytensor.com/index.php/archives/21/) · [pytorch-image-classification](https://github.com/spytensor/pytorch-image-classification)
- 个人博客：[超杰](http://spytensor.com/)
- 联系方式：zhuchaojie@buaa.edu.cn
