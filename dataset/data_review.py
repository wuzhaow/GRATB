# -*- coding: utf-8 -*-

import os
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class ReviewData(Dataset):

    def __init__(self, root_path, mode):
        if mode == 'Train':
            path = os.path.join(root_path, 'train/')
            print('loading train data')
            self.data = np.load(path + 'Train.npy', encoding='bytes')

            self.scores = np.load(path + 'Train_Score.npy')
        elif mode == 'Val':
            path = os.path.join(root_path, 'val/')
            print('loading val data')
            self.data = np.load(path + 'Val.npy', encoding='bytes')
            self.scores = np.load(path + 'Val_Score.npy')
        else:
            path = os.path.join(root_path, 'test/')
            print('loading test data')
            self.data = np.load(path + 'Test.npy', encoding='bytes')
            self.scores = np.load(path + 'Test_Score.npy')
        self.x = list(zip(self.data, self.scores))

    def __getitem__(self, idx):
        assert idx < len(self.x)

        return self.x[idx]

    def __len__(self):
        return len(self.x)
def collate_fn(batch):
        data, label = zip(*batch)
        return data, label
if __name__ == '__main__':
    # train_data=ReviewData("../dataset/Digital_Music_data",'Train')
    # train_data_loader = DataLoader(train_data, batch_size=2, shuffle=True, collate_fn=collate_fn)
    # for idx, (train_datas, scores) in enumerate(train_data_loader):
    #     print(idx)
    #     print(train_datas)
    #     print(scores)
    import numpy as np

    # 指定.npy文件路径
    # file_path = "../dataset/Digital_Music_data/train/Train.npy"
    file_path ="../dataset/Digital_Music_data/train/itemReview2Index.npy"
    # 从.npy文件加载数据
    data = np.load(file_path)

    # 显示数据
    print("Loaded data shape:", data.shape)
    print("Data content:")
    print(data)






