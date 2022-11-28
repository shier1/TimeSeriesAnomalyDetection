import logging
import torch
import random
import pickle
import numpy as np
from glob import glob
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler


logger = logging.getLogger(__name__)


class PyODDataset(Dataset):
    """
    创建数据集
    返回一个由sample数组转化的张量和整型的lable
    """
    def __init__(self, data_list, label_list):
        super(PyODDataset, self).__init__()
        self.data_list = data_list
        self.label_list = label_list

    def __len__(self):
        return self.data_list.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist() #将tensor类型转换列表格式

        sample = self.data_list[idx, :]
        label = self.label_list[idx][0]

        return torch.from_numpy(sample).float(), int(label)


def  load_data(pkl_list,label=True):
    '''
    输入pkl的列表，进行文件加载
    label=True用来加载训练集
    label=False用来加载真正的测试集，真正的测试集无标签
    '''
    X = []
    y = []
    mileage_list = []

    for  each_pkl in pkl_list:
        pic = open(each_pkl,'rb')
        item= pickle.load(pic)#下载pkl文件
        # 此处选取的是每个滑窗的最后一条数据，仅供参考，可以选择其他的方法，比如均值或者其他处理时序数据的网络
        # 取全部特征进行训练
        X.append(item[0])

        mileage = item[1]['mileage']
        mileage_list.append(mileage)
        if label:
            y.append(int(item[1]['label'][0]))
    X = np.stack(X, axis=0)
    # X = X.reshape(X.shape[0], -1)
    mileage_list = np.vstack(mileage_list)
    if label:
        y = np.vstack(y)
    return X, y, mileage_list


def get_dataset():
    """
    划分train数据集跟val数据集，并返回
    """
    train_data_path = './Train'
    train_pkl_files = glob(train_data_path+"/*.pkl")
    random.shuffle(train_pkl_files)
    train_data, train_label, train_mileage = load_data(train_pkl_files, label=True)
    train_data_max = np.max(train_data)
    train_data_min = np.min(train_data)
    train_data = (train_data - train_data_min)/(train_data_max - train_data_min)

    mileage_min = np.min(train_mileage)
    mileage_max = np.max(train_mileage)
    train_mileage = (train_mileage - mileage_min) / (mileage_max - mileage_min)
    train_mileage = np.expand_dims(train_mileage, axis=2)
    train_mileage = train_mileage.repeat(256, axis=1)

    train_data = np.concatenate([train_data, train_mileage], axis=2)

    train_dataset = PyODDataset(train_data, train_label)

    # 样本全在训练集中，此时的val_dataset, 为了兼容划分了验证集的情况
    val_dataset = PyODDataset(train_data, train_label)

    return train_dataset, val_dataset


def get_loader(args):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    trainset, valset = get_dataset()

    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler(trainset) if args.local_rank == -1 else DistributedSampler(trainset)
    val_sampler = SequentialSampler(valset)
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=args.train_batch_size,
                              num_workers=4,
                              pin_memory=True)

    val_loader = DataLoader(valset,
                             sampler=val_sampler,
                             batch_size=args.eval_batch_size,
                             num_workers=4,
                             pin_memory=True) if valset is not None else None

    return train_loader, val_loader
