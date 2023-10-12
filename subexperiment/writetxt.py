import os
import random

# 資料資料夾路徑
folder_path = '../data/mask'

# 獲取資料夾內的所有檔案名稱
file_names = os.listdir(folder_path)

# 打亂檔案順序
random.shuffle(file_names)

# 訓練集佔總檔案的比例
train_ratio = 0.8
# 驗證集佔總檔案的比例
validation_ratio = 0.1
# 測試集佔總檔案的比例
test_ratio = 0.1

# 總檔案數量
total_files = len(file_names)

# 計算劃分的索引位置
train_end = int(total_files * train_ratio)
validation_end = int(total_files * (train_ratio + validation_ratio))

# 根據索引切割檔案名稱
train_files = file_names[:train_end]
validation_files = file_names[train_end:validation_end]
test_files = file_names[validation_end:]

# 儲存到txt檔案中
def save_to_txt(file_list, file_path):
    with open(file_path, 'w') as f:
        for file in file_list:
            f.write(file + '\n')

# 儲存訓練集
save_to_txt(train_files, '../data/train.txt')
# 儲存驗證集
save_to_txt(validation_files, '../data/validation.txt')
# 儲存測試集
save_to_txt(test_files, '../data/test.txt')
