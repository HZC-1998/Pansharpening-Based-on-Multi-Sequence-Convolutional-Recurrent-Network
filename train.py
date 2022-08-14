# -*- coding: utf-8 -*

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import torch.nn as nn
from torch.nn import init
import time
import torch.optim as optim
from torch.optim import lr_scheduler
import scipy.io as sio
import numpy
from model import fusion
from visdom import Visdom
from osgeo import gdal,osr
from os.path import join
import ogr, os
from torchsummary import summary

torch.backends.cudnn.enabled = True#使用非确定性算法
torch.backends.cudnn.benchmark = True#自动寻找高效算法
## 超参数设置
version = 1  # 版本号
mav_value = 2047  # GF:1023  QB:2047
satellite = 'QB'  # gf，qb
method = 'rnn_epoch500_b64'
train_batch_size = 64
test_batch_size = 1
total_epochs = 200
test_freq = 20
model_backup_freq = 20
num_workers = 1

## 文件夹设置
testsample_dir = '../rnn-results/test-samples-v{}/'.format(version)  # 保存测试阶段G生成的图片
evalsample_dir = '../rnn-results/eval-samples-v{}/'.format(version)
record_dir = '../rnn-results/record-v{}/'.format(version)  # 保存训练阶段的损失值
model_dir = '../rnn-results/models-v{}/'.format(version)
backup_model_dir = join(model_dir, 'backup_model/')
checkpoint_model = join(model_dir, '{}-{}-model.pth'.format(satellite, method))
img_dir_train='C://Users//A//Desktop//dataset//DICNN1//WV4//train//MS_PAN_32'#输入数据集的文件路径
img_dir_target='C://Users//A//Desktop//dataset//DICNN1//WV4//train//MS_32'#目标数据集的文件路径
## 创建文件夹
if not os.path.exists(evalsample_dir):
    os.makedirs(evalsample_dir)
if not os.path.exists(testsample_dir):
    os.makedirs(testsample_dir)
if not os.path.exists(record_dir):
    os.makedirs(record_dir)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(backup_model_dir):
    os.makedirs(backup_model_dir)

## Device configuration
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('==> gpu or cpu:', device, ', how many gpus available:', torch.cuda.device_count())


def load_image(filepath):#输入数据
    img = sio.loadmat(filepath)['imgMS']  # 原始数据
    img = img.astype(np.float32)/ mav_value
    #  img = img.astype(np.float32)   # 归一化处理
    return img


class DatasetFromFolder(Dataset):#数据预加载
    def __init__(self, img_dir_train,img_dir_target ,transform=None):
        self.img_dir_train = img_dir_train
        self.input_files = os.listdir(img_dir_train)
        self.transform = transform
        self.img_dir_target=img_dir_target
        self.label_files = os.listdir(img_dir_target)
    def __len__(self):#返回数据集的长度
        return len( self.input_files )

    def __getitem__(self, index):  # idx的范围是从0到len（self）根据下标获取其中的一条数据
        input_img_path = os.path.join(self.img_dir_train, self.input_files[index])
        input_ms=load_image(input_img_path )
        target_img_path=os.path.join(self.img_dir_target, self.label_files[index])
        target=load_image(target_img_path)
        if self.transform:
            input_ms= self.transform(input_ms)
            target = self.transform(target)
        return input_ms, target


class ToTensor(object):#转变成张量
    def __call__(self, input):
        input = np.transpose(input, (2, 0, 1))
        input = torch.from_numpy(input).type(torch.FloatTensor)

        return input


def get_train_set(img_dir_train,img_dir_target):#获得训练数据集
    return DatasetFromFolder( img_dir_train,img_dir_target ,transform=transforms.Compose([ToTensor()]))


def get_test_set(testdata_dir,testdata_dir_target):#获得测试数据集
    return DatasetFromFolder(testdata_dir,testdata_dir_target,transform=transforms.Compose([ToTensor()]))


transformed_trainset = get_train_set(img_dir_train,img_dir_target)


## 训练集  ## 验证集  ## 测试集
trainset_dataloader = DataLoader(dataset=transformed_trainset, batch_size=train_batch_size, shuffle=True,
                                 num_workers=num_workers, pin_memory=True, drop_last=True)

#batch_size=一次训练所需要的最大样本
#shuffle=是否打乱训练样本的排列顺序

class DataPrefetcher():#加快数据集的读取速度
    def __init__(self, loader):
        self.loader = iter(loader)
        # self.opt = opt
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.batch = next(self.loader)
        except StopIteration:
            self.batch = None
            return

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        self.preload()
        return batch



#

criterion = nn.MSELoss(reduce=True, size_average=True).cuda()#定义损失函数

model = fusion()

model = model.cuda()


lr=0.00007
optimizer = optim.Adam(list(model.parameters()), lr=lr)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.97*lr,
                                           verbose=True, patience=2)
#SGD属于深度学习的优化方法，类似的优化方法还可以有Adam，momentum等等
if (torch.cuda.device_count() > 1):
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model,device_ids=[0, 1])#用多个GPU加速


# 模型训练
def train(model, trainset_dataloader, start_epoch):
    print('===>Begin Training!')
    model.train()#启用batch normalization和drop out。
    steps_per_epoch = len(trainset_dataloader)#训练数据个数
    total_iterations = total_epochs * steps_per_epoch#总迭代次数
    train_loss_record = open('%s/train_loss_record.txt' % record_dir, "w")
    epoch_time_record = open('%s/epoch_time_record.txt' % record_dir, "w")
    time_sum = 0
   #
   #viz = Visdom()
    #iz.line(np.array([0.]), np.array([0.]), win='pnn_train_loss', opts=dict(title='pnn_train loss'))
#绘图
    for epoch in range(start_epoch + 1, total_epochs + 1):
        start = time.time()  # 记录每轮训练的开始时刻
        train_loss = 0.0
        prefetcher = DataPrefetcher(trainset_dataloader)
        data = prefetcher.next()
        i = 0
        alphas=torch.ones(train_batch_size,5)
        while data is not None:
            i += 1
            if i >= total_epochs:
                break

            img_lr_u, target = data[0].cuda(), data[1].cuda()# cuda tensor [batchsize,C,W,H]
            alphas = alphas.float().cuda()


            train_fused_images = model(img_lr_u,alphas)  # 网络输出

            train_loss = criterion(train_fused_images, target)

            optimizer.zero_grad() # clear gradients for this training step
            train_loss.backward()   # backpropagation, compute gradients
            optimizer.step()#反向传播后参数更新

        data = prefetcher.next()
        print('=> {}-{}-Epoch[{}/{}]: train_loss: {:.15f}'.format(satellite, method, epoch, total_epochs, train_loss.item()))
        train_loss_record.write("{:.15f}\n".format(train_loss.item()))

       # viz.line(np.array([train_loss.item()]), np.array([epoch]), win='pnn_train_loss', update='append')

        # Save the model checkpoints and backup
        state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        torch.save(state, checkpoint_model)#保存模型参数，优化器参数
       # state_dict变量存放训练过程中需要学习的权重和偏执系数，state_dict作为python的字典对象将每一层的参数映射成tensor张量，需要注意的是torch.nn.Module模块中的state_dict只包含卷积层和全连接层的参数，当网络中存在batchnorm时，例如vgg网络结构，torch.nn.Module模块中的state_dict也会存放batchnorm's running_mean。

        # backup a model every epoch
        if epoch % model_backup_freq == 0:
           torch.save(model.state_dict(),
                      join(backup_model_dir, '{}-{}-model-epochs{}.pth'.format(satellite, method, epoch)))

        if epoch % test_freq == 0:
           checkpoint = torch.load(checkpoint_model)
           model.load_state_dict(checkpoint['model'])
           print('==>Testing the model after training {} epochs'.format(epoch))
          # test(model, testset_dataloader, epoch)

        # 输出每轮训练花费时间
        time_epoch = (time.time() - start)
        time_sum += time_epoch
        print('==>No:{} epoch training costs {:.4f}min'.format(epoch, time_epoch / 60))
        epoch_time_record.write(
            "{:.4f}\n".format(time_epoch / 60))

def main():
    # 如果有保存的模型，则加载模型，并在其基础上继续训练
    if os.path.exists(checkpoint_model):
        print("==> loading checkpoint '{}'".format(checkpoint_model))
        checkpoint = torch.load(checkpoint_model)
        model.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint['epoch']
        print('==> 加载 epoch {} 成功！'.format(start_epoch))
    else:
        start_epoch = 0
        print('==> 无保存模型，将从头开始训练！')

    train(model, trainset_dataloader, start_epoch)

    #eval(model, testset_dataloader)


if __name__ == '__main__':
    main()
