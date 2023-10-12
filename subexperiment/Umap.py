import umap
import numpy as np
import matplotlib.pyplot as plt
import json
import tifffile
import scipy.io as io
import scipy

import scipy.io

# Load the data
mat = scipy.io.loadmat('raw_data/CR1/CR1 slide 10__ChImJroi_DChecked_512x512__train_M__V04regp11__C50.mat')
img = mat['atlas_allcell_N']
# image_org = tifffile.imread('raw_data/CR1/CR1 slide 10_gray.tif')

mat1 = scipy.io.loadmat('raw_data/N25/N25 slide 10__ChImJroi_DChecked_512x512__train_M__V04regp11__C50.mat')
img1 = mat1['atlas_allcell_N']
# image_org1 = tifffile.imread('raw_data/N25/N25 slide 10_gray.tif')

with open('raw_data/CR1/CR1 slide 10__ChImJroi_DChecked_512x512__train_M__V04regp11__C50.json', 'r') as f:
    data = json.load(f)

info = data['info']
licenses = data['licenses']
categories = data['images']
annotations = data['annotations']

with open('raw_data/N25/N25 slide 10__ChImJroi_DChecked_512x512__train_M__V04regp11__C50.json', 'r') as f:
    data1 = json.load(f)

info1 = data1['info']
licenses1 = data1['licenses']
categories1 = data1['images']
annotations1 = data1['annotations']


num_sample = len(annotations)
num_sample1 = len(annotations1)
stacked_images = np.empty((len(annotations) + len(annotations1), 224, 224))
label = np.empty((num_sample + num_sample1, 1))

feature = np.empty((num_sample + num_sample1, 25))

class2float = {}
all_annotations = annotations + annotations1

for i in range(num_sample + num_sample1):
    class_name = all_annotations[i]['category_id3_name']
    
    if class_name not in class2float:
        class2float[class_name] = len(class2float)
    
    class_idx = class2float[class_name]
    
    label[i] = class_idx
    
    # Extract features from the current annotation
    features = [
        all_annotations[i]['CA'],
        all_annotations[i]['CHA'],
        all_annotations[i]['Extent'],
        all_annotations[i]['Density'],
        all_annotations[i]['NA'],
        all_annotations[i]['NCAr'],
        all_annotations[i]['CP'],
        all_annotations[i]['CHP'],
        all_annotations[i]['Roughness'],
        all_annotations[i]['NP'],
        all_annotations[i]['NCPr'],
        all_annotations[i]['MaxSACH'],
        all_annotations[i]['MinSACH'],
        all_annotations[i]['MajorAxisLength'],
        all_annotations[i]['MinorAxisLength'],
        all_annotations[i]['diameterBC'],
        all_annotations[i]['meanCHrd'],
        all_annotations[i]['CHSR'],
        all_annotations[i]['rMmCHr'],
        all_annotations[i]['Eccentricity'],
        all_annotations[i]['CC'],
        all_annotations[i]['CHC'],
        all_annotations[i]['FD'],
        all_annotations[i]['LC'],
        all_annotations[i]['LCstd']
    ]
    
    feature[i] = features

# for i in range(len(annotations) -1280):

#     class_name = annotations[i]['category_id3_name']
#     if class_name not in class2float:
#         class2float[class_name] = len(class2float)
#     class_name = class2float[class_name]
#     id = int(annotations[i]['id_masknii'])

#     CA = annotations[i]['CA']
#     CHA = annotations[i]['CHA']
#     Extent = annotations[i]['Extent']
#     Density = annotations[i]['Density']
#     NA = annotations[i]['NA']
#     NCAr = annotations[i]['NCAr']
#     CP = annotations[i]['CP']
#     CHP = annotations[i]['CHP']
#     Roughness = annotations[i]['Roughness'](len(annotations)+ len(annotations1)
#     NP = annotations[i]['NP']
#     NCPr = annotations[i]['NCPr']
#     MaxSACH = annotations[i]['MaxSACH']
#     MinSACH = annotations[i]['MinSACH']
#     MajorAxisLength = annotations[i]['MajorAxisLength']
#     MinorAxisLength = annotations[i]['MinorAxisLength']
#     diameterBC = annotations[i]['diameterBC']
#     meanCHrd = annotations[i]['meanCHrd']
#     CHSR = annotations[i]['CHSR']
#     rMmCHr = annotations[i]['rMmCHr']
#     Eccentricity = annotations[i]['Eccentricity']
#     CC = annotations[i]['CC']
#     CHC = annotations[i]['CHC']
#     FD = annotations[i]['FD']
#     LC = annotations[i]['LC']
#     LCstd = annotations[i]['LCstd']

#     feature[i][0] = CA
#     feature[i][1] = CHA
#     feature[i][2] = Extent
#     feature[i][3] = Density
#     feature[i][4] = NA
#     feature[i][5] = NCAr
#     feature[i][6] = CP
#     feature[i][7] = CHP 
#     feature[i][8] = Roughness
#     feature[i][9] = NP
#     feature[i][10] = NCPr
#     feature[i][11] = MaxSACH
#     feature[i][12] = MinSACH
#     feature[i][13] = MajorAxisLength
#     feature[i][14] = MinorAxisLength
#     feature[i][15] = diameterBC
#     feature[i][16] = meanCHrd
#     feature[i][17] = CHSR
#     feature[i][18] = rMmCHr
#     feature[i][19] = Eccentricity
#     feature[i][20] = CC
#     feature[i][21] = CHC
#     feature[i][22] = FD
#     feature[i][23] = LC
#     feature[i][24] = LCstd

#     # bbox = annotations[i]['bbox']
#     # y_start = bbox[0]
#     # x_start = bbox[1]
#     # y_end = bbox[2]
#     # x_end = bbox[3]

#     # img_roi = img[x_start:x_start+x_end, y_start:y_start+y_end]
#     # img_roi1 = image_org[x_start:x_start+x_end, y_start:y_start+y_end]
#     # img_clean = np.where(img_roi == id, img_roi1 , 0)



#     # padding = np.zeros((224,224))
#     # w = (224 - img_clean.shape[0]) // 2
#     # h = (224 - img_clean.shape[1]) // 2


#     # padding[w:w+img_clean.shape[0], h:h+img_clean.shape[1]] = img_clean
#     # stacked_images[i] = padding
#     label[i] = class_name
    # plt.imshow(padding)
    # plt.show()

# for i in range(len(annotations1) -1200):

#     class_name = annotations1[i]['category_id3_name']
#     if class_name not in class2float:
#         class2float[class_name] = len(class2float)
#     class_name = class2float[class_name]
#     id = int(annotations1[i]['id_masknii'])
#     # bbox = annotations1[i]['bbox']
#     # y_start = bbox[0]
#     # x_start = bbox[1]
#     # y_end = bbox[2]
#     # x_end = bbox[3]

#     # img_roi = img1[x_start:x_start+x_end, y_start:y_start+y_end]
#     # img_roi1 = image_org1[x_start:x_start+x_end, y_start:y_start+y_end]

#     # img_clean = np.where(img_roi == id, img_roi1 , 0)

#     # padding = np.zeros((224,224))
#     # w = (224 - img_clean.shape[0]) // 2
#     # h = (224 - img_clean.shape[1]) // 2


#     # padding[w:w+img_clean.shape[0], h:h+img_clean.shape[1]] = img_clean

#     # stacked_images[i+len(annotations)] = padding

#     print(i)
#     CA = annotations[i]['CA']
#     print(CA)
#     CHA = annotations[i]['CHA']
#     print(CHA)
#     Extent = annotations[i]['Extent']
#     Density = annotations[i]['Density']
#     NA = annotations[i]['NA']
#     NCAr = annotations[i]['NCAr']
#     CP = annotations[i]['CP']
#     CHP = annotations[i]['CHP']
#     Roughness = annotations[i]['Roughness']
#     NP = annotations[i]['NP']
#     NCPr = annotations[i]['NCPr']
#     MaxSACH = annotations[i]['MaxSACH']
#     MinSACH = annotations[i]['MinSACH']
#     MajorAxisLength = annotations[i]['MajorAxisLength']
#     MinorAxisLength = annotations[i]['MinorAxisLength']
#     diameterBC = annotations[i]['diameterBC']
#     meanCHrd = annotations[i]['meanCHrd']
#     CHSR = annotations[i]['CHSR']
#     rMmCHr = annotations[i]['rMmCHr']
#     Eccentricity = annotations[i]['Eccentricity']
#     CC = annotations[i]['CC']
#     CHC = annotations[i]['CHC']
#     FD = annotations[i]['FD']
#     LC = annotations[i]['LC']
#     LCstd = annotations[i]['LCstd']

#     feature[i][0] = CA
#     feature[i][1] = CHA
#     feature[i][2] = Extent
#     feature[i][3] = Density
#     feature[i][4] = NA
#     feature[i][5] = NCAr
#     feature[i][6] = CP
#     feature[i][7] = CHP 
#     feature[i][8] = Roughness
#     feature[i][9] = NP
#     feature[i][10] = NCPr
#     feature[i][11] = MaxSACH
#     feature[i][12] = MinSACH
#     feature[i][13] = MajorAxisLength
#     feature[i][14] = MinorAxisLength
#     feature[i][15] = diameterBC
#     feature[i][16] = meanCHrd
#     feature[i][17] = CHSR
#     feature[i][18] = rMmCHr
#     feature[i][19] = Eccentricity
#     feature[i][20] = CC
#     feature[i][21] = CHC
#     feature[i][22] = FD
#     feature[i][23] = LC
#     feature[i][24] = LCstd

#     label[i+len(annotations)] = class_name


# # print(stacked_images.shape)

flatten_feature = feature.reshape(num_sample + num_sample1, -1).astype(np.float32)
print(flatten_feature.shape)



# 转换数据类型为 float32
# flatten_feature[flatten_feature > np.finfo(np.float32).max] = np.finfo(np.float32).max
# flatten_feature = flatten_feature.astype(np.float32)

# print("NaN Count:", np.isnan(flatten_feature).sum())
# print("Max Value:", np.max(flatten_feature))
# print("Min Value:", np.min(flatten_feature))


# from sklearn.preprocessing import RobustScaler

# scaler = RobustScaler()
# flatten_feature = scaler.fit_transform(flatten_feature)




# 截断或标准化异常值


# # Generate a random dataset with 100 samples and 10 features
# # data = np.random.rand(100, 10)



# # # Create a UMAP model with desired parameters
from mpl_toolkits.mplot3d import Axes3D  # 导入绘制三维图的库

# 创建 UMAP 模型，将 n_components 设置为 3
reducer = umap.UMAP(n_neighbors=3, min_dist=0.1, n_components=2)

from umap import utils
# 对展平后的图像数据进行降维
# disconnected_points = utils.disconnected_vertices(flatten_feature)
# print("Disconnected Points:", disconnected_points)

# 删除不连通的点

labels = label.flatten()
print(labels.shape)

# 对处理后的数据运行 UMAP
embedding = reducer.fit_transform(flatten_feature)



# 获取不同类别的唯一值
unique_labels = np.unique(labels)

# 创建一个颜色映射，以便为每个类别选择不同的颜色
colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))

# 创建一个三维散点图来显示 UMAP 的结果，不同类别使用不同颜色
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')  # 创建一个三维坐标系

# for i, class_label in enumerate(unique_labels):
#     mask = labels == class_label

#     class_org_name = next((key for key, value in class2float.items() if value == int(class_label)), None)

#     ax.scatter(embedding[mask, 0], embedding[mask, 1], embedding[mask, 2], label=f'Class {class_org_name}', c=colors[i])

# ax.set_title('UMAP Projection in 3D')
# ax.legend()
# plt.show()

fig, ax = plt.subplots(figsize=(10, 8))

for i, class_label in enumerate(unique_labels):
    mask = labels == class_label

    class_org_name = next((key for key, value in class2float.items() if value == int(class_label)), None)

    ax.scatter(embedding[mask, 0], embedding[mask, 1], label=f'Class {class_org_name}', c=colors[i])

ax.set_title('UMAP Projection in 2D')
ax.legend()
plt.show()




