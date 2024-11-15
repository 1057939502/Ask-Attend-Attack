import pandas as pd

# 读取txt文件
data = pd.read_csv(r'F:\dataset\Flick30k\captions.txt')

# 将数据保存为csv文件
data.to_csv(r'F:\dataset\Flick30k\captions.csv', index=False)
