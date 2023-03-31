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

mmcv

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