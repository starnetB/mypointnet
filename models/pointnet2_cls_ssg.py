import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstraction


class get_model(nn.Module):
    def __init__(self,num_class,normal_channel=True):
        super(get_model, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_class)

    def forward(self, xyz):
        B, _, _ = xyz.shape  # xyz.shape [b,d,n]
        if self.normal_channel:
            norm = xyz[:, 3:, :]  #需要保留通道吗(b,d,n)
            xyz = xyz[:, :3, :]  #[b,c,n]
        else:
            norm = None
        # xyz ,[B,c=3,n=1024]
        # norm  [B,d=3,n=1024]
        # group_all:sample=>centroid => [B,1,3] &&[B,1,n=1024,3/6]  
        # group:sample=>centorid => [B,S=512,3] &&[B,S=512,32,3/6]
        #[B,3,512] [B,128,512] or group_all:[B,3,1] [B,128,1]
        l1_xyz, l1_points = self.sa1(xyz, norm) 
        # l1_xyz=[B,3,512]  or [B,3,1]
        # l1_points=[B,128,512] or [B,128,1]
        # group_all:sample=centorid=> [B,1,3] && [B,1,1,3+(128)]
        # group:sample=>centorid =>[B,128,3] && [B,128,64,3+(128)]
        # conv [3+(128),128,1] =>[128,128,1]=>[128,256,1]
        # [B,3,128] [B,256,128] or [B,3,1] [B,256,1] 
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        # l2_xyz=[B,3,128] 
        # l2_points=[B,256,128] 
        # group_all:sample=centrod =>[B,1,3] && [B,1,128,3+256]
        # conv [3+256,256,1]  [256,512,1] [512,1024,1]
        # [B,1024,128,1]
        # max=>[B,1024,1]
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)  
        x = l3_points.view(B, 1024)  # x=[B,1024]  #去掉最后一个维度
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))  # 全链接层[1024,512]
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))  # 全连接层[512,256]
        x = self.fc3(x)  #[256,class]
        x = F.log_softmax(x, -1)  #进行log[softmax]


        return x, l3_points  #[B,num_class(-log(softmax))]  [B,1024,1]



class get_loss(nn.Module):
    def __init__(self):        
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        # nll_closs [-log(id)]
        # [0.1,0.2,0.3]  targ[1] => [-0.2]
        total_loss = F.nll_loss(pred, target)

        return total_loss
