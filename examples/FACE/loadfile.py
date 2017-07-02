import numpy as np
import pandas as pd
import os
import sys


train_file = 'training.csv'
test_file = 'test.csv'

def load(test=False, cols=None):
    """
    载入数据，通过参数控制载入训练集还是测试集，并筛选特征列
    """
    fname = test_file if test else train_file
    df = pd.read_csv(os.path.expanduser(fname))

    # 将图像数据转换为数组
    df['Image'] = df['Image'].apply(lambda x: np.fromstring(x, sep=' '))

    # 筛选指定的数据列
    if cols:  
        df = df[list(cols) + ['Image']]

    print(df.count())  # 每列的简单统计
    df = df.dropna()  # 删除空数据

    # 归一化到0到1
    X = np.vstack(df['Image'].values) / 255.  
    X = X.astype(np.float32)

    # 针对训练集目标标签进行归一化
    if not test:
        y = df[df.columns[:-1]].values
        y = (y - 48) / 48  
        X, y = shuffle(X, y, random_state=42)
        y = y.astype(np.float32)
    else:
        y = None

    return X, y