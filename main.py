from AtomModel import *


# 从Vasp文件夹中准备数据集，并将其用ANItransform转化，得到编码后的向量存储到pkl文件中
def prepare_data_set(vasp_dir_path):
    aim_vasp_path = vasp_dir_path # "S:\数据集\碳纳米管掺杂\\5-5\\b\oh"

    print_file(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>New Game Begin!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    print_file("Start Data Collecting")

    dataset_maker = DatasetMaker(aim_vasp_path)
    dataset_maker.make_dataset()
    total_info = dataset_maker.give_out_dataset()
    print_file("Finished Data Collecting, Start Feature Transform")

    # 坐标编码
    dataset_offer = DatasetOffer(total_data_info=total_info)
    total_train_feed_x, \
    total_test_feed_x, \
    total_train_feed_y, \
    total_test_feed_y, \
    atom_cases, \
    n_feat = \
        dataset_offer.ANI_transform(save_pkl_path="ANI_features.pkl")

# 将ANI编码过后的feature用于训练，存储模型到/model中
def load_pkl_file_to_train(ANI_pkl_file_path="ANI_features.pkl",epoch=100):

    nn = FullAtomModel(atom_cases, os.getcwd() + "/model", n_feat)
    with open(ANI_pkl_file_path, "rb") as f:
        # _代表不使用train的数据
        total_train_feed_x, _, total_train_feed_y, _, atom_cases, n_feat\
            = pickle.load(f)
    try:
        nn.load_atom_weights()
    except:
        print_file("Load Weights Failed")

    string = 'Total feed X shape: \n'
    string += "Train: "

    for i in total_train_feed_x:
        for j in i:
            print(i[j])
            string += j + ":" + str(i[j].shape)
    string += "\n"

    string += "Test: "
    for i in total_test_feed_x:
        for j in i:
            string += j + ":" + str(i[j].shape)
    string += "\n"
    print_file(string)


    index = 0
    for i in range(epoch):# 这里用while True也行，因为每次fit都会保存weights
        #print_file(">>Loop %s/%s"%(i+1,repeat))
        print_file(">>Loop %s"%(index))

        for dataset_index in range(len(total_train_feed_x)):
            print_file(">>>>Train for %s/%s" % (dataset_index+1, len(total_train_feed_x)))
            nn.fit(total_train_feed_x[dataset_index],total_train_feed_y[dataset_index],epoch=1000,load_weights=True)
        index += 1

    nn.save_atom_weights()


def predict(ANI_pkl_file_path="ANI_features.pkl"):

    # 直接使用编码后的数据，也可以现进行编码
    with open(ANI_pkl_file_path, "rb") as f:
        # _代表不使用train的数据
        _, total_test_feed_x, _, total_test_feed_y, atom_cases, n_feat\
            = pickle.load(f)

    # 读取存储的模型
    nn = FullAtomModel(atom_cases, os.getcwd() + "/model/trained", n_feat)

    nn.load_atom_weights()

    pred_result = []
    true_result = []
    for dataset_index in range(len(total_test_feed_x)):
        pred_result.extend(nn.predict(total_test_feed_x[dataset_index]))
        true_result.extend(total_test_feed_y[dataset_index])


    plt.plot(pred_result,true_result,'ro')
    plt.savefig("test_result.png",dpi=300)
    plt.show()



prepare_data_set("/public/home/yangbo1/wangbch/nanotube/9-9")