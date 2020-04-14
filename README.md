### 声明：开源只是为了方便大家交流学习，数据请勿用于商业用途！！！！转载或解读请注明出处，谢谢！

**背景**

很早之前开源过 pytorch 进行图像分类的代码（[从实例掌握 pytorch 进行图像分类](http://spytensor.com/index.php/archives/21/)），历时两个多月的学习和总结，近期也做了升级。在此基础上写了一个 Ai Challenger 农作物竞赛的 baseline 供大家交流。

**2018 年 12 月 13 日更新**

新增数据集下载链接：[百度网盘]( https://pan.baidu.com/s/16f1nQchS-zBtzSWn9Guyyg ) 提取码：iksk 
数据集是 10 月 23 日 更新后的新数据集，包含训练集、验证集、测试集A/B.
另外最近有同学拿到类似的数据，想做分类的任务，但是这份代码是针对这次比赛开源的，在数据读取方式上会有区别，对于新手来说不太友好，我开源了一份针对图像分类任务的代码，并附上简单教程，相信看完后能比较轻松使用 pytorch 进行图像分类。

教程: [从实例掌握 pytorch 进行图像分类](http://www.spytensor.com/index.php/archives/21/)

代码: [pytorch-image-classification](https://github.com/spytensor/pytorch-image-classification)

**2018年 10 月 30 日更新**

新增 `data_aug.py` 用于线下数据增强，由于时间问题，这个比赛不再做啦，这些增强方式大家有需要可以研究一下，支持的增强方式：

- 高斯噪声
- 亮度变化
- 左右翻转
- 上下翻转
- 色彩抖动
- 对比度变化
- 锐度变化

注：对比度增强在可视化后，主观感觉特征更明显了，目前我还未跑完。提醒一下，如果做了对比度增强，在测试集的时候最好也做一下。

个人博客：[超杰](http://spytensor.com/)

比赛地址：[农作物病害检测](https://challenger.ai/competition/pdr2018)

完整代码地址：[plants_disease_detection](https://github.com/spytensor/plants_disease_detection)

    注：
    欢迎大佬学习交流啊，这份代码可改进的地方太多了,
    如果大佬们有啥改进的意见请指导！
    联系方式：zhuchaojie@buaa.edu.cn

**成绩**：线上 0.8805，线下0.875，由于划分存在随机性，可能复现会出现波动，已经尽可能排除随机种子的干扰了。

## 提醒

`main.py` 中的test函数已经修正，执行后在 `./submit/`中会得到提交格式的 json 文件,现已支持 Focalloss 和交叉验证，需要的自行修改一下就可以了。
依赖中的 pytorch 版本请保持一致，不然可能会有一些小 BUG。

### 1. 依赖

    python3.6 pytorch0.4.1

### 2. 关于数据的处理

首先说明，使用的数据为官方更新后的数据，并做了一个统计分析（下文会给出），最后决定删除第 44 类和第 45 类。
并且由于数据分布的原因，我将 train 和 val 数据集合并后，采用随机划分。

数据增强方式：

- RandomRotation(30)
- RandomHorizontalFlip()
- RandomVerticalFlip()
- RandomAffine(45)

图片尺寸选择了 650，暂时没有对这个尺寸进行调优（毕竟太忙了。。）

### 3. 模型选择

模型目前就尝试了 resnet50，后续有卡的话再说吧。。。

### 4. 超参数设置

详情在 config.py 中

### 5.使用方法

- 第一步：将测试集图片复制到 `data/test/` 下
- 第二步：将训练集合验证集中的图片都复制到 `data/temp/images/` 下，将两个 `json` 文件放到 `data/temp/labels/` 下
- 执行 move.py 文件
- 执行 main.py 进行训练

### 6.数据分布图

训练集

![train](http://www.spytensor.com/images/plants/train.png)

验证集

![val](http://www.spytensor.com/images/plants/val.png)

全部数据集

![all](http://www.spytensor.com/images/plants/all.png)
