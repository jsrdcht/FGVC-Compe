<img src="resources/mmcls-logo.png" width="600"/>
  <div>&nbsp;</div>
  <div align="center">
    <b><font size="5">OpenMMLab 官网</font></b>
    <sup>
      <a href="https://openmmlab.com">
        <i><font size="4">HOT</font></i>
      </a>
    </sup>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <b><font size="5">OpenMMLab 开放平台</font></b>
    <sup>
      <a href="https://platform.openmmlab.com">
        <i><font size="4">TRY IT OUT</font></i>
      </a>
    </sup>
  </div>
  <div>&nbsp;</div>


# 环境准备
mmcls 1.x

mmengine

mmcv 1.x

# start
1. 先用这个脚本设置训练集文件夹结构
```python
import pandas as pd
import os
import shutil
from tqdm import tqdm

# 读取 train.csv 文件
train_data = pd.read_csv("train.csv")

# 获取所有唯一的 label
unique_labels = train_data["label"].unique()

# 在 iBioHash_Train 下创建 label 对应的子文件夹
train_data_folder = "iBioHash_Train"
for label in unique_labels:
    os.makedirs(os.path.join(train_data_folder, str(label)), exist_ok=True)

# 遍历 train.csv 中的每一行，并将对应的数据移动到相应的子文件夹下
for _, row in tqdm(train_data.iterrows(), total=train_data.shape[0], desc="移动文件"):
    image_id = row["image_id"]
    label = row["label"]
    source_path = os.path.join(train_data_folder, image_id)
    destination_path = os.path.join(train_data_folder, str(label), image_id)

    # 检查源文件是否存在，然后移动文件
    if os.path.exists(source_path):
        shutil.move(source_path, destination_path)
    else:
        print(f"文件 {source_path} 不存在.")
```

2. 然后启动脚本在tools/train.sh

```shell
bash tools/train.sh
```

3. 在生成hash_code之前，需要先生成query.csv和gallery.csv。注意调整内置的参数，环境不一定相同

```shell
python tools/gen_ann.py
```

4. 生成hash_code的脚本在tools/gen_hash_code.py，注意调整参数

```shell
python tools/gen_hash_code.py
```

5. 生成hash_code之后，使用官方的脚本生成最终提交结果

```shell
python generate_submit_csv.py
```

# 实验记录

- **2023-4-1 1:43:00 log->mmclassification/results/test/20230331_123649** - 实现了Greedyhash算法，但是不能收敛。在消融后（去除greedyhash loss保留分类loss）发现是hash_layer的问题，生成hash_code后无法简单地完成分类。在hash_feature做了tanh()变换，然后再做sign()，变得好训了很多。acc从10%上涨到了50%。

  现在的问题是，训练20epoch后止步在了50%左右，无法继续提升。可能是触及到了hash_code的分类性能上限。

  目前计划：1. 调整超参，加入greedyhash loss，训练greedy hash最终版30epoch。

- **2023-4-1 3:50:00 log->mmclassification/results/test/20230401_035603** - 加入greedyhash loss后发生了不收敛的情况，再次检查代码。发现是loss写错了，mm框架里每个loss不需要手动加，框架会自动汇总。
- **2023-4-3 1:00:00 log->home/ct/code/fgvc/mmclassification/results/beit2_greedy/20230403_003656** - 上一次实验成功收敛，但是acc只有50%。我认为是因为hash_code做分类的极限可能就是这么多了。生成了hash_code并提交，得分并不高。下一步的改进方向是：我看到排行榜有个人正确率超出其它人一大截，如果只是小幅度使用trick是赶不上他的分数的。需要思考是不是哪个地方做的不对，找出这个地方才能涨点比较多。主要的思考方向是：1. 算法 2. domain shift 目前在稍微调整模型看看使用哪个模型好一点，确认模型之后再调整算法。

# 算法知识

- **GreedyHash** 算法是一种用于大规模数据的近似最近邻搜索（Approximate Nearest Neighbor Search）的方法。它利用学习得到的哈希函数将高维数据映射到低维哈希空间，使得在哈希空间中具有相似性的数据点距离接近。在这样的映射过程中，我们希望哈希编码具有良好的性质，例如均匀分布、高度离散化等。

  在网络中加入greedy_loss = (hash_feature.abs() - 1).pow(3).abs().mean()的作用是为了让学习到的哈希特征（hash_feature）具有更好的性质。具体来说：

  对哈希特征取绝对值（hash_feature.abs()），是为了确保特征值非负。这样做可以使得在计算哈希特征距离时，计算量较小且易于处理。

  然后减去1（(hash_feature.abs() - 1)），目的是让哈希特征值接近1。在GreedyHash中，我们希望哈希特征的值尽量为1，以便生成二值化哈希编码（+1 或 -1）。

  将结果三次方（.pow(3)），使得特征值偏离1的惩罚更为严重，进一步增强了特征值接近1的约束力度。

  再次取绝对值（.abs()），是为了将负数映射为正数，这样计算损失时，大的损失值对应于哈希特征与1相差较大。

  最后求平均（.mean()），计算损失值。这个损失值可以作为优化目标，用于优化网络参数，以生成更好的哈希特征。

  通过引入greedy_loss，我们鼓励模型生成接近1的哈希特征值。这有助于生成具有更好性质的哈希编码，从而提高近似最近邻搜索的性能。

