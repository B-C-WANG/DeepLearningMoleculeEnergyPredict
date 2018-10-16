# -*- coding: utf-8 -*-

import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from deepchem.data  import NumpyDataset
import tensorflow as tf
import deepchem as dc
import time
import os
from DatasetMaker import *
from VDE.AtomSpace import atom_index_trans_reverse

import matplotlib.pyplot as plt
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

# TODO: 这个文件要改名

def print_file(string):
    with open("log", "a") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S> ", time.localtime())+string+'\n')




class DatasetOffer(object):

    def __init__(self,total_data_info=None,dataset_pkl_path=None):
        if total_data_info == None and dataset_pkl_path == None:
            raise ValueError("Should not be both None of total data nor dataset_pkl_path")
        if total_data_info is not None and dataset_pkl_path is not None:
            raise ValueError("Offer only one of data info and dataset_pkl_path")
        if dataset_pkl_path is not None:
            with open(dataset_pkl_path, "rb") as f:
                self.total_info = pickle.load(f)
                print_file("read pkl finished")
            if self.total_info["instance"] != 'DatasetMaker':
                raise ValueError("Input data should generated from DatasetMaker")
        else:
            self.total_info = total_data_info

        self.aim_sample_keys = None

    def filter_dataset(self):
        # filt dataset according to some condition
        self.aim_sample_keys = list(self.total_info.keys())# all data include, no filter
        self.aim_sample_keys.remove("instance")


    def generate_dataset(self,test_size):
        if self.aim_sample_keys == None:
            self.filter_dataset()

        train_datasets = []
        test_datasets = []
        # 对于每一个文件，打乱里面的数据然后分训练和测试集，每个里面都是所有原子坐标构成的帧
        for i in self.aim_sample_keys:

            #print(self.total_info[i])
            #print("A",self.total_info[i]["atom_cases"])

            trainX,testX,trainy,testy = train_test_split(self.total_info[i]['x'],
                                                         self.total_info[i]['y'],
                                                         test_size=test_size,shuffle=True)
            train_datasets.append(NumpyDataset(X=trainX,y=trainy))
            test_datasets.append(NumpyDataset(X=testX,y=testy))
            atom_cases = self.total_info[i]["atom_cases"] # 随便来一个atom cases就行，因为一批训练是一样的
        return train_datasets, test_datasets, atom_cases

    def ANI_transform(self,save_pkl_path=False):
        '''
        in : list of data(sample number, atom number, 3)
        out list of data(sample number, atom number, feature_num)
        :return:

        由于每种原子要feed进入不同的NN，所以用atom index作为key，对应的feature作为value

        TODO: 将字典改成用类进行feed
        稍微改一下增加灵活性，目前用字典feed，最好用类feed
        一个vasp sample是一个batch，这是由于矩阵大小不固定，固定大小的矩阵才是一个batch
        另外纵观全局，要做到：既能一行代码从vasp坐标到NN训练，又要每一步都能单独拿出提取，和feature转换后的pkl文件

        '''

        train_datasets, test_datasets, atom_cases = self.generate_dataset(test_size=0.2)
        # 这个分割是把每个数据集里面分出X 和 Y，但是似乎应当将所有数据集混合在一起然后再分更为合理，
        # 或者在训练时混合不同来源的数据集
        # TODO：这里数据集没有扩增，不同来源的数据集样本比例有差异！



        total_train_feed_x = []
        total_test_feed_x = []

        total_train_feed_y = []
        total_test_feed_y = []
        dataset_number = len(train_datasets)
        print("Dataset Number: ",dataset_number)
        #dataset_number = 2 # 减少计算量使用
        for dataset_index in range(dataset_number): # 这里每个文件进行处理

            # 对于每一个文件样本，里面归类成一个dict
            nn_train_x = {}
            nn_test_x = {}
            for i in atom_cases:
                nn_test_x[atom_index_trans_reverse[int(i)]] = []
                nn_train_x[atom_index_trans_reverse[int(i)]] = []

            print_file("processing %s" % dataset_index)

            train_dataset = train_datasets[dataset_index]
            test_dataset = test_datasets[dataset_index]

            total_train_feed_y.append(train_dataset.y)
            total_test_feed_y.append(test_dataset.y)


            max_atom_number = train_dataset.X.shape[1]
            #print(atom_cases)

            r_feature_num = 32
            a_feature_num = 8

            transformer = dc.trans.ANITransformer(
                max_atoms=max_atom_number,
                atom_cases=list(atom_cases),
                radial_length= r_feature_num,
                angular_length=a_feature_num,
                atomic_number_differentiated=True,


                # 这里建议是True，这样可以向量按照原子分开，每一个原子是一个向量，而不是先给每个原子的radial向量再给angular向量
                # the shape is : sample number, atom number, feature (which is atom_index + bond_feature + angle_feature)
                # take 4 atom type for example:
                # this feature is 1 + 128 + 640, from tensorboard, we have 4, so should be 1 + 32*4 + 160*4
                # transfer 128 to 32 32 32 32, 640 to 160 160 160 160, and then concat to 32+160 * 4, the 4 go to different NN
                # 128 = 4 * 32
                # 160 = 10 * 64
                # 这里要转换， 要把 32 * 4 + 160 * 4转换为 32+160 * 4，这样feature向量可以独立开来到不同神经网络
                # 一般来说一批训练的atom cases是一样的，所以转换的feature也是一样的
                # 先将feature数目除以原子数目n，因为是n个原子连续拼接成的向量，之后减去32，因为半径对称函数固定是32！具体看ANITransformer构造函数
                # 角度对称函数feature是8，具体看ANITransformer构造函数，这里乘以了20，20与


                # 要注意，这里的原子要最大化成体系，比如CHONi，如果有CHNi而没有O的，也要把O的feature加进去，否则feature的大小不如预期
)

            # 如果batch_size过大，可能造成OOM
            transformer.transform_batch_size = 2
            atom_cases_num = len(atom_cases)

            trans_train = transformer.transform(train_dataset)
            trans_test = transformer.transform(test_dataset)

            trainX = trans_train.X
            testX = trans_test.X

            # 这里的转化把r feature和a feature连在一起，以对应原子为group，便于今后分类，但可能是不必要的
            # trainX, testX = tran_features(trainX,testX)
            def tran_features(trainX,testX):
                def x_trans(input):
                    atom_index = input[:,:,0]
                    features = input[:,:,1:]

                    # 这里有问题，不应该先分成4份，应该先把半径 feature提取出来，再划分和concat
                    features_r = features[:,:,:atom_cases_num*r_feature_num]
                    features_a = features[:,:,atom_cases_num*r_feature_num:]
                    # 分成4份，注意4应该是倒数第二个，最后一个才是feature
                    features_r = features_r.reshape(features_r.shape[0],
                                                    features_r.shape[1],
                                                    atom_cases_num,
                                                    r_feature_num,)
                    features_a = features_a.reshape(features_a.shape[0],
                                                    features_a.shape[1],
                                                    atom_cases_num,
                                                    features_a.shape[2] // atom_cases_num,)
                    features = np.concatenate([features_r,features_a],axis=3)
                    features = features.reshape(features.shape[0],features.shape[1],-1)

                    features = np.insert(features,0,values=atom_index,axis=2)# 最后把index加上

                    return features
                trans_trainX = x_trans(trainX)
                trans_testX = x_trans(testX)
                #这里check一下feature的提取是否正确
                def plot_out_feature():
                    plt.plot(trainX[0,20,:],"r",alpha=0.5)
                    #print(trainX.shape)
                    plt.plot(trans_trainX[0,20,:],"b",alpha=0.5)
                    #print(trans_trainX.shape)
                    plt.show()
                plot_out_feature()
                return trans_trainX, trans_test






            n_feat = transformer.get_num_feats()
            #print("info>>>>>>>>>>>>>>\n")
            
            #print(n_feat)

            # 转化之后将其归类，用于输出， 其中原子序数的feature保留，并且除以118来归一化
            def group_by_atom(output_dict,X):
                # 数据集有的原子，
                exist_atom = X[0, :, 0].astype(np.int)
                exist_atom = set(exist_atom)

                for atom_index in atom_cases:
                    # 如果没有这种原子，那就把feature全部搞成0，但是这样有可能受到bias的加和不为零的影响

                    if atom_index not in exist_atom:
                        output_dict[atom_index_trans_reverse[int(atom_index)]] = np.zeros(shape=(X.shape[0],1,n_feat))
                        continue

                    one_atom_feature = []
                    for i in range(X.shape[0]):
                        i_sample = X[i, :, :]

                        one_sample_feature = []
                        atom_exist = 0
                        for j in range(i_sample.shape[0]):
                            if int(i_sample[j, 0]) == atom_index:
                                f = [i_sample[j, 0] / 118]
                                one_sample_feature.append(np.concatenate([f,i_sample[j, 1:]],axis=0))
                                atom_exist = 1

                        one_atom_feature.append(np.expand_dims(np.array(one_sample_feature),axis=0))

                    output_dict[atom_index_trans_reverse[int(atom_index)]] = np.concatenate(one_atom_feature,axis=0)



            group_by_atom(nn_train_x,trainX)
            group_by_atom(nn_test_x,testX)




            total_train_feed_x.append(nn_train_x)
            total_test_feed_x.append(nn_test_x)

        new_atom_cases = []
        for i in atom_cases:
            new_atom_cases.append(atom_index_trans_reverse[i])
        atom_cases = new_atom_cases


        if save_pkl_path:
            with open(save_pkl_path, "wb") as f:
                pickle.dump([total_train_feed_x,total_test_feed_x,total_train_feed_y,total_test_feed_y,atom_cases, n_feat],f)


        return total_train_feed_x,total_test_feed_x,total_train_feed_y,total_test_feed_y,atom_cases, n_feat


def from_pkl():
    test =  DatasetOffer(dataset_pkl_path="C:\\Users\Administrator\Desktop\\dataset.pkl")
    test.ANI_transform(save_pkl_path="C:\\Users\Administrator\Desktop\\feature.pkl")

def from_datasetMaker():
    dataset = DatasetMaker("C:\\Users\Administrator\Desktop\Pt\AllSurfaceG1")
    dataset.make_dataset()
    dataset.save_dataset("C:\\Users\Administrator\Desktop\\dataset.pkl")
    test = DatasetOffer(total_data_info=dataset.give_out_dataset())
    test.ANI_transform()




if __name__ == '__main__':
    from_pkl()





