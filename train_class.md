# pointNet 网络结构简介

# 分类部分
输入数据: [batch_size,1024,6]   
输入标签: [batch_size,]

## 结构概要
* Batch_Size:B
* Channel:C
* nSample:欧式距离，最近点的数量，他包括radius和nSample(最大采样点个数)
* Feature:当前点云，每个中心点，欧式距离最近的nSample个点的训练出的特征
* PointNetSetAbstriction1=>input_xyz\[Batch_size,3,1024](B,C,N)::input_norml[Batch_size,3,1024](B,C,N)  
    * Sample:group(cetroid=512,nsample=32)  =>[Batch_size,512,32,6](B,N,nSample,Feature)
    * MLP(CNN)(inchannel=3,mlp=[64, 64, 128])=>[Batch_size,128,32,512](B,Feature,nSample,N)
    * max in nsample()=>[Batch_size,128,512](B,Feature,N)
* PointNetSetAbstriction2=>input_xyz[Batch_size,3,512](B,C,N)::input_norml[Batch_size,128,512](B,C,N)
    * Sample:group(cetroid=128,nsample=64)  =>[Batch_size,128,64,128+3](B,N,nSample,Feature)
    * MLP(CNN)(inchannel=3+128,mlp=[128,128,256])=>[Batch_size,256,64,128](B,Feature,nSample,N)
    * max in nsample=>[Batch_size,256,128](B,Feature,N)
* PointNetSetAbstriction3=>input_xyz[Batch_size,3,128](B,C,N)::input_norml[Batch_size,256,128](B,C,N)
    * Sample:group_all=>[Batch_size,1,128,256+3](B,N,nSample,Feature)
    * MLP(CNN)(inchannel=256+3,mlp=[256, 512, 1024])=>[Batch_size,1024,128,1](B,Feature,nSample,N)
    * max in nsample=>[Batch_size,1024,1](B,Feature,N)
* view(-1)
* fc_bn_drop_relu1(1024,512)=>[Batch_size,512](B,Feature)
* fc_bn_drop_relu2(512,256)=>[Batch_size,256](B,Feature)
* fc_bn_drop_relu3(256,num_class)=>[Batch_size,num_class](B,Feature(class))

## 输入处理
points=[batch_size,6,1024]    

* 点云的xyz坐标(最远采样的方法，获取1024个点)
point_xyz=[batch_size,3,1024]
* 点云的法向量坐标(最远采样的方法，获取1024个点)
point_norm=[batch_size,3,1024]

## 网路结构详解
PointNet++分类网络最为主要的部分是PointNetSetAbstraction，它由如下部分组成，我们可以这部分为Encoder    
* sample采样部分，这部分有两个情况，一是Group_all，另外一种是Group
    * Group:对当前点云中包含的点进行最远采样，选出nPoint个中心点(如1024->512),然后在原来的点云上计算里这nPoint个中心点最近距离的nSample点(如1024(原来点云的数量),512(中心点数量))那么最终的结果就是[Batch_size,npoint(512),nSample(32),()],但这里还缺少了一个我们需要的特征，填入()中的内容，便是每个点包含的特征，结果根据前面的内容进行输入。如上面的输入特征数量是xyz(3),normal(3)那么结果就是6,第二层的话就是3+128,最终的结果就是[Batch_size,npoint(512),nSample(32),(6)],当然返回的结果中还需要包含中心点，那么就是[Batch_size,npoint(512),3](3)
    * Group_all:这就比较简单了，首先创建一个全是0的伪中心点坐标，[B,1,C=3]=>[0]，然后将原来的输入进行调整,比如上面第一层的输入就是[B,1,N=1024,C+D=6]
* mlp部分：这里其实就是卷积层
    * 首先对Tensor进行处理，如上面第一层的表现形式如下所示[Batch_size,npoint(512),nSample(32),Feature(6)]，那么我们要处理的是通道部分我们对Tensor进行调整[Batch_size,Feature(6),nSample(32,npoint(512))]
    * 然后我们使用卷积分对Feature进行调整，还是上面第一层PointNetSetAbstraction为例子:
        * conv(in_channel=6,64,1)=>conv(64,64,1)=>conv(64,64,1)=>conv(64,128,1)=>输出[Batch_size,Feature(128),nSample(32),npoint(512))]
* max层
    * 在nSample进行max压缩
    * 第一层为例：[Batch_size,Feature(128),nSample(32),npoint(512)]=>max(1)=>[Batch_size,Feature(128),npoint(512)]

经过三层的PointNetSetAbstraction的调整之后，结果就会变成[Batch_size,1024,1](B,Feature,N)    
然后review一下就变成了[Batch_size,1024]
* 然后通过几层全链接层+bn+Dropout+relu之后就变成
* [Batch_size,num_class]


## 损失函数
输入:[Batch_size,num_class]
Label:[Batch_size,]

在num_class计算log_softmax:    
然后在计算nll_loss:    
结果公式就是如下形式-log(softmax(class_id))，最大化这个就可以了。