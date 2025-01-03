import logging
import math
import pdb
import numpy as np
import torch
import random
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.datasets import EMNIST
# from .datasets import EMNIST_truncated
from torch.utils.data import DataLoader, Dataset

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):

        image, label = self.dataset[int(self.idxs[item])]
        return image, label

def split_train_test(dataset, idxs, batch_size):
    # split train, and test
    # idxs_train = idxs
    train = DataLoader(DatasetSplit(dataset, idxs), batch_size=batch_size, shuffle=True)
    return train


def _data_transforms_emnist():
    # CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    # CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        # transforms.RandomResizedCrop(size=(200, 200), scale=(0.1, 1), ratio=(0.5, 2)),
        transforms.RandomGrayscale(),
        transforms.ToTensor(),
        # transforms.Normalize((0.1307,), (0.3081,)),
        
    ])

    # train_transform.transforms.append(Cutout(16))

    valid_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        # transforms.RandomResizedCrop(size=(200, 200), scale=(0.1, 1), ratio=(0.5, 2)),
        transforms.RandomGrayscale(),
        transforms.ToTensor(),
        # transforms.Normalize((0.1307,), (0.3081,)),
    ])

    return train_transform, valid_transform

def dirichlet_cifar_noniid(degree_noniid, dataset, num_users):
    
    train_labels = np.array(dataset.targets)
    num_classes = len(dataset.classes)
    
    label_distribution = np.random.dirichlet([degree_noniid]*num_users, num_classes)
    
    # print(label_distribution)
    # print(sum(label_distribution), sum(np.transpose(label_distribution)), sum(sum(label_distribution)))
    
    class_idcs = [np.argwhere(train_labels==y).flatten() for y in range(num_classes)]
    
    dict_users = [[] for _ in range(num_users)]
    for c, fracs in zip(class_idcs, label_distribution):
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1]*len(c)).astype(int))):
            dict_users[i] += [idcs]

    # print(dict_users, np.shape(dict_users))
    
    dict_users = [set(np.concatenate(idcs)) for idcs in dict_users]
    
    return dict_users

def pat_cifar_noniid(degree_noniid, dataset, num_users):
    
    train_labels = np.array(dataset.targets)
    num_classes = len(dataset.classes)
    
    label_distribution = np.random.dirichlet([degree_noniid]*num_users, num_classes)
    # label_distribution = 
    
    print(label_distribution)
    print(sum(label_distribution), sum(np.transpose(label_distribution)), sum(sum(label_distribution)))
    
    class_idcs = [np.argwhere(train_labels==y).flatten() for y in range(num_classes)]
    
    dict_users = [[] for _ in range(num_users)]
    for c, fracs in zip(class_idcs, label_distribution):
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1]*len(c)).astype(int))):
            dict_users[i] += [idcs]

    # print(dict_users, np.shape(dict_users))
    
    dict_users = [set(np.concatenate(idcs)) for idcs in dict_users]
    
    return dict_users

def load_partition_data_emnist( data_dir, partition_method, partition_alpha, client_number, batch_size, logger):
    transform_train, transform_test = _data_transforms_emnist()
    # dataset_train = EMNIST(data_dir, train=True, download=False,
    #                     transform=transform_train, split = 'letters')
    # dataset_test = EMNIST(data_dir, train=False, download=False,
    #                     transform=transform_test, split = 'letters')

    dataset_train = EMNIST(data_dir, train=True, download=True,
                        transform=transform_train, split = 'letters')
    dataset_test = EMNIST(data_dir, train=False, download=True,
                        transform=transform_test, split = 'letters')
    if partition_method=="dir":
        dict_train = dirichlet_cifar_noniid(partition_alpha, dataset_train, client_number) # non-iid
        dict_test = dirichlet_cifar_noniid(partition_alpha, dataset_test, client_number) # non-iid\
    else:
        dict_train = pat_cifar_noniid(partition_alpha, dataset_train, client_number) # non-iid
        dict_test = pat_cifar_noniid(partition_alpha, dataset_test, client_number) # non-iid\



    data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()

    for client_idx in range(client_number):
        local_data = dict_train[client_idx] 
        data_local_num_dict[client_idx] = len(local_data)
        train_data_local_dict[client_idx] = split_train_test(dataset_train, list(dict_train[client_idx]),batch_size)
        # test_data_local_dict[client_idx]  = DataLoader(DatasetSplit(dataset_test ,  dict_test[client_idx]), batch_size=len(dict_test[client_idx]), shuffle=False)
        test_data_local_dict[client_idx]  = split_train_test(dataset_test , list(dict_test[client_idx]), batch_size)
        # split_train_test(dataset_test , list(dict_test[client_idx]),batch_size)

    return data_local_num_dict, train_data_local_dict, test_data_local_dict




    # X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
    #                                                                                          data_dir,
    #                                                                                          partition_method,
    #                                                                                          client_number,
    #                                                                                          partition_alpha, logger)
    # # get local dataset
    # data_local_num_dict = dict()
    # train_data_local_dict = dict()
    # test_data_local_dict = dict()
    # transform_train, transform_test = _data_transforms_emnist()
    # cache_train_data_set= EMNIST(data_dir, train=True, transform=transform_train, download=True, split = 'letters' )
    # print("....")
    # cache_test_data_set = EMNIST(data_dir, train=False, transform=transform_test, download=True, split = 'letters' )
    # idx_test = [[] for i in range(62)]
    # # checking
    # for label in range(62):
    #     idx_test[label] = np.where(y_test == label)[0]
    # test_dataidxs = [[] for i in range(client_number)]
    # tmp_tst_num = math.ceil(len(cache_test_data_set) / client_number)
    # for client_idx in range(client_number):
    #     for label in range(62):
    #         # each has 620 pieces of testing data
    #         label_num = math.ceil(traindata_cls_counts[client_idx][label] / sum(traindata_cls_counts[client_idx]) * tmp_tst_num)
    #         rand_perm = np.random.permutation(len(idx_test[label]))
    #         if len(test_dataidxs[client_idx]) == 0:
    #             test_dataidxs[client_idx] = idx_test[label][rand_perm[:label_num]]
    #         else:
    #             test_dataidxs[client_idx] = np.concatenate(
    #                 (test_dataidxs[client_idx], idx_test[label][rand_perm[:label_num]]))
    #     dataidxs = net_dataidx_map[client_idx]
    #     train_data_local, test_data_local = get_dataloader_emnist( data_dir, batch_size, batch_size,
    #                                              dataidxs,test_dataidxs[client_idx] ,cache_train_data_set=cache_train_data_set,cache_test_data_set=cache_test_data_set,logger=logger)
    #     local_data_num = len(train_data_local.dataset)
    #     data_local_num_dict[client_idx] = local_data_num
    #     logger.info("client_idx = %d, local_sample_number = %d" % (client_idx, local_data_num))
    #     train_data_local_dict[client_idx] = train_data_local
    #     test_data_local_dict[client_idx] = test_data_local

    # record_part(y_test, traindata_cls_counts, test_dataidxs, logger)

    # return None, None, None, None, \
    #        data_local_num_dict, train_data_local_dict, test_data_local_dict, traindata_cls_counts
