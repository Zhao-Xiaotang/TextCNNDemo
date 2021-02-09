"""
    对数据进行预处理，将所给数据集完成划分
"""
import os
import pandas as pd
import numpy as np

random_seed = 100

# 设置数据集路径
Dir = './data/'
origin_data_path = os.path.join(Dir, '2021MCMProblemC_DataSet.csv')
label_path = os.path.join(Dir,'Detection/labels.csv')
train_path = os.path.join(Dir,'Detection/train.csv')
test_path = os.path.join(Dir,'Detection/test.csv')
dev_path = os.path.join(Dir,'Detection/dev.csv')

# 对初始数据进行处理，得到分类数据集
COLUMN_NAMES = ['GlobalID', 'Detection Date', 'Notes', 'Lab Status',	'Lab Comments', 'Submission Date', 'Latitude', 'Longitude']
df_origin_detect = pd.read_csv(origin_data_path, names=COLUMN_NAMES, header=0)
df_origin_data = df_origin_detect[['Lab Status','Notes']]

# 分离数据
label_list = ['Positive ID', 'Negative ID', 'Unverified', 'Unprocessed']
train_list = []
test_list = []
origin_data = np.array(df_origin_data)
count = 0

for i in range(4440):
    if origin_data[i,0] in ['Positive ID', 'Negative ID']:
        if origin_data[i,0] == 'Positive ID':
            train_list.append(np.insert(origin_data[i], 0, [0]))
        else:
            train_list.append(np.insert(origin_data[i], 0, [1]))
    else:
        test_list.append(np.insert(origin_data[i], 0, count))
        count +=1
        pass
    pass

name_train = ['Label ID', 'Label', 'Notes']
name_test = ['ID', 'Label', 'Notes']
df_train = pd.DataFrame(columns=name_train, data=train_list)
df_test = pd.DataFrame(columns=name_test,data=test_list)

# 写入文件
if __name__ == '__main__':
    df_train.to_csv(train_path)
    df_test.to_csv(test_path)





