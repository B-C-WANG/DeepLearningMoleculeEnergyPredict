from dlmep.AtomModel import FullAtomModel
import numpy as np
import pickle
from dlmep.DatasetOffer import DatasetOffer,print_file
from dlmep.DatasetMaker import DatasetMaker

import traceback


# 从Vasp文件夹中准备数据集，并将其用ANItransform转化，得到编码后的向量存储到pkl文件中
def prepare_data_set(vasp_dir_path):
    aim_vasp_path = vasp_dir_path  # "S:\数据集\碳纳米管掺杂\\5-5\\b\oh"

    print_file(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>New Game Begin!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    print_file("Start Data Collecting")

    dataset_maker = DatasetMaker(aim_vasp_path)
    dataset_maker.make_dataset()
    total_info = dataset_maker.give_out_dataset()
    print_file("Finished Data Collecting, Start Feature Transform")
    print(total_info)
    # 坐标编码
    dataset_offer = DatasetOffer(total_data_info=total_info)
    total_train_feed_x, \
    total_test_feed_x, \
    total_train_feed_y, \
    total_test_feed_y, \
    atom_cases, \
    n_feat = \
        dataset_offer.ANI_transform(save_pkl_path="ANI_features.pkl")
    print(total_train_feed_x)
    print(total_train_feed_x.key)
    exit()
    print(total_train_feed_x)

# 将ANI编码过后的feature用于训练，存储模型到/model中
def load_pkl_file_to_train(ANI_pkl_file_path="ANI_features.pkl", epoch=100):
    with open(ANI_pkl_file_path, "rb") as f:
        # _代表不使用train的数据
        total_train_feed_x, total_test_feed_x, total_train_feed_y, total_test_feed_y, atom_cases, n_feat \
            = pickle.load(f)
    nn = FullAtomModel(atom_cases, os.getcwd() + "/model", n_feat)
    try:
        nn.load_atom_weights()
    except:
        print_file("Load Weights Failed")

    string = 'Total feed X shape: \n'
    string += "Train: "

    for i in total_train_feed_x:
        string += "\n"
        for j in i:
            print(i[j])
            string += j + ":" + str(i[j].shape)
    string += "\n"

    string += "Test: "
    for i in total_test_feed_x:
        string += "\n"
        for j in i:
            string += j + ":" + str(i[j].shape)
    string += "\n"
    print_file(string)



    def train_group_by_dataset_from():  # 根据数据来源（vasp文件夹）划分，训练时容易过拟合
        index = 0
        for i in range(epoch):  # 这里用while True也行，因为每次fit都会保存weights
            # print_file(">>Loop %s/%s"%(i+1,repeat))
            print_file(">>Loop %s" % (index))

            for dataset_index in range(len(total_train_feed_x)):
                print_file(">>>>Train for %s/%s" % (dataset_index + 1, len(total_train_feed_x)))
                nn.fit(total_train_feed_x[dataset_index], total_train_feed_y[dataset_index], epoch=1000,
                       load_weights=True)
            index += 1

    def train_shuffle_by_batch_size(batch_size=32):
        X = {}
        y = np.array([])
        # key随便来源一个数据集
        atom_cases = list(total_train_feed_x[0].keys())
        print(atom_cases)
        for dataset_index in range(len(total_train_feed_x)):
            print("Dataset have samples: %s" % len(total_train_feed_y[dataset_index]))
            '''
            total_train_feed_x是一个List
            每个例如以O C H 等元素为Key，之后为样本数量，原子数目和Feature长度！所以应当分别截取各个元素的前batchsize个
            
            
            
            '''

            # 进行解包数据聚合，需要内存较大，如果样本来源很多需要修改处理方法
            # 如果原子数目不一样，不能使用这个方法或者需要进行改进
            for atom in atom_cases:
                try:
                    X[atom] = np.concatenate([total_train_feed_x[dataset_index][atom], X[atom]], axis=0)
                except KeyError:
                    X[atom] = total_train_feed_x[dataset_index][atom]

            y = np.concatenate([y,total_train_feed_y[dataset_index].reshape(-1)])

        print("Y: ",y.shape)
        print("X: ")
        for atom in atom_cases:
            print(X[atom].shape)# 这里需要和前面输出的total data sample一起看是否有遗漏

        # 进行shuffle
        shuffle_index = np.array(list(range(y.shape[0])))
        print(np.random.shuffle(shuffle_index))

        for atom in atom_cases:
            X[atom] = X[atom][shuffle_index]
        print("Before Shuffle: ",y[:10])
        y = y[shuffle_index]
        print("After shuffle: ",y[:10])
        sample_numbers = y.shape[0]


        # 开始重新分批次，每次一个batchsize
        batch_num = int(sample_numbers / batch_size)+1
        newX = {}
        newY = {}
        '''
        这里和之前的数据结构不同了，之前是list，现在是dict
        
        '''
        for i in range(batch_num-1):

            for atom in atom_cases:
                try:
                    newX[i][atom] = X[atom][i*batch_size:1*batch_size+batch_size]
                except:
                    newX[i] = {}
                    newX[i][atom] = X[atom][i*batch_size:1*batch_size+batch_size]

            newY[i] = y[i*batch_size:1*batch_size+batch_size]
        # 还剩下一个残余的batch
        newX[batch_num-1] = X[atom][(batch_num-1)*batch_size:]
        newY[batch_num-1] = y[batch_num*batch_size:]

        # 剩下的训练就和之前的一样了
        X = newX
        y = newY
        # 这里再显示一下信息

        index = 0
        while 1:
            index += 1
            print_file(">>Loop %s" % (index))
            for dataset_index in X:
                print(X[dataset_index])
                print(y[dataset_index])
                nn.fit(X[dataset_index], y[dataset_index], epoch=1000,
                       load_weights=True)



    train_group_by_dataset_from()
    #train_shuffle_by_batch_size()

    nn.save_atom_weights()


def predict(ANI_pkl_file_path="ANI_features.pkl"):
    # 直接使用编码后的数据，也可以现进行编码
    with open(ANI_pkl_file_path, "rb") as f:
        # _代表不使用train的数据
        _, total_test_feed_x, _, total_test_feed_y, atom_cases, n_feat \
            = pickle.load(f)

    # 读取存储的模型
    nn = FullAtomModel(atom_cases, os.getcwd() + "/model/trained", n_feat)

    nn.load_atom_weights()

    pred_result = []
    true_result = []
    for dataset_index in range(len(total_test_feed_x)):
        pred_result.extend(nn.predict(total_test_feed_x[dataset_index]))
        true_result.extend(total_test_feed_y[dataset_index])

    plt.plot(pred_result, true_result, 'ro')
    plt.savefig("test_result.png", dpi=300)
    plt.show()


#prepare_data_set("/public/home/yangbo1/wangbch/nanotube/9-9")
prepare_data_set("S:\FTP\数据集\碳纳米管掺杂\整理后\\7-7\ooh")
load_pkl_file_to_train()
