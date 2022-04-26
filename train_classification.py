"""
Author: Benny
Date: Nov 2019
"""

import os
import sys
import torch
import numpy as np

import datetime
import logging
import provider
import importlib
import shutil
import argparse

from pathlib import Path
from tqdm import tqdm
from data_utils.ModelNetDataLoader import ModelNetDataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    # pointnet++ 还是pointnet
    parser.add_argument('--model', default='pointnet_cls', help='model name [default: pointnet_cls]')
    # 分类数量40/10
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40],  help='training on ModelNet10/40')
    # 训练周期
    parser.add_argument('--epoch', default=200, type=int, help='number of epoch in training')
    # 
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    # 一副图默认点的最大数量
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    # 优化器,默认adam
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    # 日志路径
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    # 权重衰减
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    # 是否使用正则化
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    # 处理数据的位置
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    # 是否使用统一采样
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    return parser.parse_args()


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True


def test(model, loader, num_class=40):
    mean_correct = []  #平均正确率
    class_acc = np.zeros((num_class, 3)) # 分类精度
    classifier = model.eval()  # 评估模式

    for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):

        if not args.use_cpu:
            points, target = points.cuda(), target.cuda()

        points = points.transpose(2, 1)  #[B,6,npoint=1024]
        pred, _ = classifier(points)  #[B,log(num_class softmax)]  [B,1024,1]
        pred_choice = pred.data.max(1)[1]  #[B] 求最大值的索引

        for cat in np.unique(target.cpu()):  # 把target中的类一个一个拿出来进行比较
            classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0]) #每个batch_size,
            class_acc[cat, 1] += 1   #一个epoch中含有几个batch

        correct = pred_choice.eq(target.long().data).cpu().sum()  #一个batch的整体正确率
        mean_correct.append(correct.item() / float(points.size()[0]))  #和

    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_acc = np.mean(class_acc[:, 2])  # 先求各个类别正确率，再将类别正确率/batch 的和除以batch，最后对类别求平均
    instance_acc = np.mean(mean_correct) # 平均正确率

    return instance_acc, class_acc  #整理平均正确率，(class_acc先求类别正确率，在类别的基础上求平均)


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('classification')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        # 日志里面添加时间
        exp_dir = exp_dir.joinpath(timestr)
    else:
        # 添加制定路径点
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    # 检查点
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    # 创建log
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    data_path = 'data/modelnet40_normal_resampled/'

    train_dataset = ModelNetDataLoader(root=data_path, args=args, split='train', process_data=args.process_data)
    test_dataset = ModelNetDataLoader(root=data_path, args=args, split='test', process_data=args.process_data)
    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)
    # [batch_size,npoint=1024,D=6 ]  labels [batch_size,1]
    '''MODEL LOADING'''
    
    num_class = args.num_category
    #从model中导入pointernet2_cls_ssg
    model = importlib.import_module(args.model)
    #将三个模型文件copy到指定路径下面
    shutil.copy('./models/%s.py' % args.model, str(exp_dir))
    shutil.copy('models/pointnet2_utils.py', str(exp_dir))
    shutil.copy('./train_classification.py', str(exp_dir))

    # 得到分类器
    classifier = model.get_model(num_class, normal_channel=args.use_normals)
    #家在损失函数
    criterion = model.get_loss()
    classifier.apply(inplace_relu)

    if not args.use_cpu:
        classifier = classifier.cuda()
        criterion = criterion.cuda()

    try:
        # 加载之前保存的模型
        checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth')
        # 获取检查点的起点
        start_epoch = checkpoint['epoch']
        # 加载模型参数
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        # 没有加载成功的话，要从零开始
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0
    # Adam
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999), 
            eps=1e-08,  #收敛条件
            # 增加权重衰减
            weight_decay=args.decay_rate
        )
    else:
        # 使用SGD方法
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    #每20步降低学习率
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0
   
    '''TRANING'''
    
    logger.info('Start training...')
    for epoch in range(start_epoch, args.epoch):
        #训练到第几个周期
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        mean_correct = []
        classifier = classifier.train()  #修改为训练模式，使能，dropout和bn等

        scheduler.step() #迭代一次权重,每20步需要降低到0.7*lr
        # enumerate(可迭代数据集本身，start=0)
        for batch_id, (points, target) in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()

            points = points.data.numpy() #数据转nompy
            points = provider.random_point_dropout(points) # 随机去出一部分数据
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3]) #随机缩放
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])  #随机平移
            points = torch.Tensor(points)  #tensor话  [batch_size,npoint,6]
            points = points.transpose(2, 1)   #[batch_size,6,npoint]

            if not args.use_cpu:
                points, target = points.cuda(), target.cuda()

            pred, trans_feat = classifier(points)  #[B,num_class(log(softmax))]  [B,1024,1]
            loss = criterion(pred, target.long(), trans_feat)  #[-log(softmax(id))]
            pred_choice = pred.data.max(1)[1]   #[B,num_class]=>[B] 把预测结果拿出来

            correct = pred_choice.eq(target.long().data).cpu().sum()  #把预测结果中正确的做一个sum
            mean_correct.append(correct.item() / float(points.size()[0]))  #平均
            loss.backward()  #反向求导
            optimizer.step()  #梯度
            global_step += 1 #总的迭代次数

        train_instance_acc = np.mean(mean_correct)  #求平均精度 训练平均进度
        log_string('Train Instance Accuracy: %f' % train_instance_acc)

        #下面球测试平均进度
        with torch.no_grad():
            # classofier.eval() 转为测试模式，防止bn dropout等操作
            instance_acc, class_acc = test(classifier.eval(), testDataLoader, num_class=num_class)

            if (instance_acc >= best_instance_acc):
                best_instance_acc = instance_acc
                best_epoch = epoch + 1

            if (class_acc >= best_class_acc):
                best_class_acc = class_acc
            log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))
            log_string('Best Instance Accuracy: %f, Class Accuracy: %f' % (best_instance_acc, best_class_acc))

            if (instance_acc >= best_instance_acc):
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': best_epoch,
                    'instance_acc': instance_acc,
                    'class_acc': class_acc,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
            global_epoch += 1

    logger.info('End of training...')


if __name__ == '__main__':
    args = parse_args()
    main(args)
