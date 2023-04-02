import os

# 设置图片文件夹路径
image_dir = "/home/ct/code/fgvc/iBioHash/Gallery"

# 准备输出的标注文件
output_file = "gallery_ann.txt"

# 检查并获取所有jpg图片
image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

# 创建并打开标注文件
with open(output_file, 'w') as ann_file:
    # 遍历所有jpg图片
    for image_file in image_files:
        # 只保留"Query"文件夹部分的图片路径
        image_path = os.path.join("Gallery", image_file)
        # 将图片路径和标签0写入标注文件
        ann_file.write(f"{image_path} 0\n")

print(f"已成功创建标注文件：{output_file}")

