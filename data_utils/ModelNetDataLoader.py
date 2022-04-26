'''
@author: Xu Yan
@file: ModelNet.py
@time: 2021/3/19 15:51
'''
import os

from sqlalchemy import true
import numpy as np
import warnings
import pickle

from tqdm import tqdm
from torch.utils.data import Dataset

warnings.filterwarnings('ignore')


def pc_normalize(pc):
    #对点云进行正则化
    centroid = np.mean(pc, axis=0)
    # 去平均值
    pc = pc - centroid
    # 归一化
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape  #N,D=3 or ther6
    xyz = point[:,:3]   #去出xyz坐标
    centroids = np.zeros((npoint,))  #[npoint=1024,]的列表 0
    distance = np.ones((N,)) * 1e10  #[n]的列表 1e10  n掉表一个点云中点的总数
    farthest = np.random.randint(0, N) #[0,N]之间的随机数 
    # 随机选择一个点，放到0号位置，
    # 然后当前点，与所有其他点计算距离，形成距离列表，使用相对小的值更新distance
    # 然后，选取出距离最大的Farthest中
    # 使用Farthest作为1号位置的点
    # 重复2，3过程，知道所有点都给选择完毕npoint个点
    for i in range(npoint):   # 开始随机采点，并开始去中心化
        centroids[i] = farthest  # [npoint=1024]列表中的没一个点，代表这原先
        centroid = xyz[farthest, :]    #[x,y,z]
        dist = np.sum((xyz - centroid) ** 2, -1)  #[N]的列表 没一个代表当前点，与你随机选择的中心点之间的距离
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    # 把这些点重新拿出来，然后再次赋予一个点
    point = point[centroids.astype(np.int32)]

    return point


class ModelNetDataLoader(Dataset):
    def __init__(self, root, args, split='train', process_data=False):
        self.root = root
        self.npoints = args.num_point
        self.process_data = process_data
        self.uniform = args.use_uniform_sample
        self.use_normals = args.use_normals
        self.num_category = args.num_category

        if self.num_category == 10:
            # 加载10分类的对象数据
            self.catfile = os.path.join(self.root, 'modelnet10_shape_names.txt')
        else:
            # 加载40分类的对象数据
            self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')
        # 删除前后空格，然后变成list
        self.cat = [line.rstrip() for line in open(self.catfile)]
        # dict=["class",id]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        shape_ids = {}
        if self.num_category == 10:
            # 点云数据集文件名[txt1,txt2.....]
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_test.txt'))]
        else:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]

        assert (split == 'train' or split == 'test')
        #代表对应的名字
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        #print(shape_ids['train'][0:10])
        #print(shape_names[0:10])
        #ids:['airplane_0001', 'airplane_0002', 'airplane_0003', 'airplane_0004', 'airplane_0005', 'airplane_0006', 'airplane_0007', 'airplane_0008', 'airplane_0009', 'airplane_0010']
        #name: ['airplane', 'airplane', 'airplane', 'airplane', 'airplane', 'airplane', 'airplane', 'airplane', 'airplane', 'airplane']
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                         in range(len(shape_ids[split]))]
        #print(self.datapath[0:2])
        #[('airplane', 'data/modelnet40_normal_resampled/airplane/airplane_0627.txt'), ('airplane', 'data/modelnet40_normal_resampled/airplane/airplane_0628.txt')]
        print('The size of %s data is %d' % (split, len(self.datapath)))
    
        # 保存路径 什么意思
        if self.uniform:
            self.save_path = os.path.join(root, 'modelnet%d_%s_%dpts_fps.dat' % (self.num_category, split, self.npoints))
        else:
            self.save_path = os.path.join(root, 'modelnet%d_%s_%dpts.dat' % (self.num_category, split, self.npoints))
     
      
        if self.process_data:
            if not os.path.exists(self.save_path):
                print('Processing data %s (only running in the first time)...' % self.save_path)
                self.list_of_points = [None] * len(self.datapath)
                # print([None]*3)
                # 输出话点云列表:[None,None,None]，每个None应该就是一个点云
                self.list_of_labels = [None] * len(self.datapath)
                # 输出话一个标签列表，每个None应该就是一个标签

                # 就是一个循环函数，但是可以添加进度条
                for index in tqdm(range(len(self.datapath)), total=len(self.datapath)):
                #for index in range(len(self.datapath)):
                    # 一个元组
                    # fn：('airplane', 'data/modelnet40_normal_resampled/airplane/airplane_0627.txt')
                    fn = self.datapath[index]
                    # 类便，并且通过classes变成对应的id
                    cls = self.classes[self.datapath[index][0]]
                    # 变成np:[id]
                    cls = np.array([cls]).astype(np.int32)
                    # 加载点云集合
                    point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

                    if self.uniform:
                        #
                        point_set = farthest_point_sample(point_set, self.npoints)
                    else:
                        # 如果不需要统一采样的花直接截取前面的npoints的内容
                        point_set = point_set[0:self.npoints, :]

                    self.list_of_points[index] = point_set  
                    self.list_of_labels[index] = cls
                    # 其实就是选点的方法,每个点集选择1024个点

                with open(self.save_path, 'wb') as f:
                    pickle.dump([self.list_of_points, self.list_of_labels], f)
            else:
                print('Load processed data from %s...' % self.save_path)
                with open(self.save_path, 'rb') as f:
                    self.list_of_points, self.list_of_labels = pickle.load(f)
            # self.list_of_points   [full_size,nPoint,D]
            # self.list_of_lables   [full_size,1,1]

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        if self.process_data:
            #如果需要处理数据，就直接从list_of_points[index], self.list_of_labels[index]
            #pointset [npoint,D]
            #lable   [1,1]
            point_set, label = self.list_of_points[index], self.list_of_labels[index]
        else:
            #如果不需要处理数据的话，我们直接处理，并且不保存
            fn = self.datapath[index]
            cls = self.classes[self.datapath[index][0]]
            label = np.array([cls]).astype(np.int32)
            point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

            if self.uniform:
                point_set = farthest_point_sample(point_set, self.npoints)
            else:
                point_set = point_set[0:self.npoints, :]
                
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])  #去对xyz去均值，然后归一化
        if not self.use_normals:
            point_set = point_set[:, 0:3]

        return point_set, label[0]

    def __getitem__(self, index):
        return self._get_item(index)


if __name__ == '__main__':
    import torch

    data = ModelNetDataLoader('/data/modelnet40_normal_resampled/', split='train')
    DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)
    for point, label in DataLoader:
        print(point.shape)
        print(label.shape)
