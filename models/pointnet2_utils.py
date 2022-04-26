import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np

def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]  #[]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)  #[B,npoint=512]=>  [0] 512 个中心点
    distance = torch.ones(B, N).to(device) * 1e10   #[B,N=1024] =>1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device) #[B] =>rand[0,N-1]
    batch_indices = torch.arange(B, dtype=torch.long).to(device)  #[0,B-1]
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape  #[B,N=1024,C=3]
    _, S, _ = new_xyz.shape  #[S=512]
    # [0,N-1] =>[1,1,N] =>[B,S,N]
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz) #[计算点点之间的欧式距离] [B,S=512,c=3]  [B,N=1024,c=3] =[B,S,N]
    group_idx[sqrdists > radius ** 2] = N  # 将那些距离比搜索半径大的点，置为最后孙需的点
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]  #去出，对点的距离进行排序，去出最大点nsample, [B,S,nsampe] 
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])   # 将第一组去出扩展，也就是顺序最考前的那一个片，切割，然后复制
    mask = group_idx == N  # 也就是最后超出最大范围的点，最后都有最近点的集合替换
    group_idx[mask] = group_first[mask]  # 对才超出半径范围的点，进行替换
    return group_idx  #[B,S,nsample]  #距离点


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape #[B,N,3]
    S = npoint    #[512]
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint=512]
    new_xyz = index_points(xyz, fps_idx)    #·[B, S=512, C=3]
    idx = query_ball_point(radius, nsample, xyz, new_xyz)  #[B,S=512,nsample]
    grouped_xyz = index_points(xyz, idx) # [B, S, nsample, C=3]  #利用距离，和最大点nsample ,吧最近的nsample个点的内容找出来
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)   #所有点减去中心点的坐标，彻底的变成局部信息

    if points is not None:  # 如果存在法向量的话，我们就使用法向量进行cat操作
        grouped_points = index_points(points, idx)  #[B.S.nsample,3] 提炼出法向量信息
        # [B, npoint, nsample, C+D=6] # c=3 包含nsample个离中心点距离最近点的信息，并且已经去中心化了，D=3就是法向量信息
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) 
    else:
        new_points = grouped_xyz_norm  #[B,S,nsample,3]  没有发响亮信息
    if returnfps:
        # new_xyz,中心点坐标  [B,S=512,3]
        # new_points，[可能带有法向量的 局部信息]  [B,S,nsample,3+3]
        # grouped_xyz,  [不带有法向量的 局部信息，只含有xyz，并且没有去中心话] [B,S,nample,3]
        # 中心点距离采样信息·fps_idx=[B,npoint=512]
         return new_xyz, new_points, grouped_xyz, fps_idx  # 
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape  #[B,N,3]
    new_xyz = torch.zeros(B, 1, C).to(device) #[B,1,3] =0
    grouped_xyz = xyz.view(B, 1, N, C)  #[B,N,C]=>[B,1,N,3]  
    if points is not None:  # 如果有法向量特征
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)  #[B,1,N,3+D=3]
    else:  #如果没有法向量特征
        new_points = grouped_xyz #[B,1,N,C]
    return new_xyz, new_points  #[B,1,3]  [B,1,N=1024,3+D=3]


class PointNetSetAbstraction(nn.Module):
    # npoint=512, radius=0.2, nsample=32, in_channel=in_channel(3,6), mlp=[64, 64, 128], group_all=False)
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint  # 512
        self.radius = radius  # k临近半径
        self.nsample = nsample  # k临近数量
        self.mlp_convs = nn.ModuleList()  # 多层感知机列表
        self.mlp_bns = nn.ModuleList()  # 多层感知机bns
        last_channel = in_channel  # 最后的通道
        for out_channel in mlp: # 多少个mlp
            # [batch_size,3,npoint=512]
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            # [batch_size,output,npoint=512]
            last_channel = out_channel
        self.group_all = group_all #[64,96,128]

    #
    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C=3, N]
            points: input points data, [B, D=3, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)  #xyz  [B;N;3]
        if points is not None:   
            points = points.permute(0, 2, 1)  #[B,N,3]

        if self.group_all:  #
            #[B,1,3]=>0 ,[B,1,N=1024,3+D]
            # 这里打包所有点，也就是不对点，进行删除
            # 这应该应该是对待稠密点云的方法
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            # 这里的npoint已经不是前面的npoint了，注意哦
            # 之前是1024,这里第一部分是512
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        # 如果是group_all,也就是整体处理的,
        #  new_xyz,[B,1,3] 并且全是0
        #  new_points=[B,1,N=1024,3] 这个是针对稀疏矩阵的你应该能理解吗
        # 否则 new_xyz=[B,npoint,c] 就是中心点坐标
        #     new_xyz=[B,npoint,nsample,C+D]
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            #[64, 64, 128]  conv(3,64,1) bn relu,conv(64,64,1),bn relu, conv(64,128,1),bn relu
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points))) 
        
        # [B,128,nsample,npoint=s=512]  or [B,128,1024,1]

        new_points = torch.max(new_points, 2)[0]  # 在采样最近距离的nsample 维度上取得最大值，每个中心点上的最大值 [B,128,512] or [B,128,1]
        new_xyz = new_xyz.permute(0, 2, 1)  #[B,c=3,npoint=512] or [B,c=3,npoint=1]
        return new_xyz, new_points  


class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint #1024
        self.radius_list = radius_list  # 半径列表，
        self.nsample_list = nsample_list # 采样列表,
        self.conv_blocks = nn.ModuleList()  # 卷积模块列表
        self.bn_blocks = nn.ModuleList()  # bn模块
        for i in range(len(mlp_list)):  # mlp列表
            convs = nn.ModuleList()  #卷积列表初始化
            bns = nn.ModuleList()  #正则化层初始化
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:   # 第一层
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
        S = self.npoint
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points =  F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points

