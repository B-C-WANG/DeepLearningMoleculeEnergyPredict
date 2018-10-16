# -*- coding:utf-8 -*-
import tensorflow as tf
import os
import copy
import pickle
import numpy as np
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from DatasetMaker import *
from DatasetOffer import *




class AtomModel(object):
    '''
    每个原子的网络
    输入 样本数目None，同种原子数目None，Feature数目固定
    输出为样本数目的能量None 1，中间全连接+reduce sum


    CH训练的网络不能迁移到CHO，但CHO的可以迁移到CH，所以以更大的为准来编码和训练，更大的atom cases

    '''



    def __init__(self,feature_num,atom):

        self.feature_num = feature_num
        self.atom = atom
        self.build_graph()


    def build_graph(self):
        # 输入为样本数量，同种原子数量，feature_num
        # scope为原子名称
        with tf.variable_scope('%s'% self.atom) :
            # 先定义正常输出和全0输出
            def return_zero_output():
                # 这里直接加和而不是输出zero，因为zero需要有固定shape # 输入 NoneA 1 NoneB
                a = tf.reduce_sum(self.input,axis=2)
                return a

            def build_valid_output():
                hidden = tf.contrib.layers.fully_connected(
                    self.input,
                    num_outputs=128,
                    activation_fn=tf.nn.tanh,
                    biases_initializer=tf.truncated_normal_initializer,
                    weights_initializer=tf.truncated_normal_initializer,
                    trainable=True
                )

                hidden = tf.contrib.layers.fully_connected(
                    hidden,
                    num_outputs=64,
                    activation_fn=tf.nn.tanh,
                    biases_initializer=tf.truncated_normal_initializer,
                    weights_initializer=tf.truncated_normal_initializer,
                    trainable=True
                )
                # 下面是新添加的两层
                hidden = tf.contrib.layers.fully_connected(
                    hidden,
                    num_outputs=32,
                    activation_fn=tf.nn.tanh,
                    biases_initializer=tf.truncated_normal_initializer,
                    weights_initializer=tf.truncated_normal_initializer,
                    trainable=True,
                )

                hidden = tf.contrib.layers.fully_connected(
                    hidden,
                    num_outputs=16,
                    activation_fn=tf.nn.tanh,
                    biases_initializer=tf.truncated_normal_initializer,
                    weights_initializer=tf.truncated_normal_initializer,
                    trainable=True,
                )
                # 上面是新添加的两层


                output = tf.contrib.layers.fully_connected(
                    hidden,
                    num_outputs=1,
                    activation_fn=None, # 这里很重要，如果不为None采用其他max为1的激活函数，如果只有40个原子最大输出为40，能量为几百就根本预测不到
                    biases_initializer=tf.truncated_normal_initializer,
                    weights_initializer=tf.truncated_normal_initializer,
                    trainable=True)
                # 输出为样本数量的该种原子能量
                self.output = tf.reduce_sum(output,axis=1)
                return self.output

            self.input = tf.placeholder(shape=[None,None,self.feature_num],dtype="float32")
            # 作一个判断，如果全部是0的feature，和不可能大于0.0001，直接输出0
            self.output = tf.cond(pred=tf.less(tf.reduce_sum(self.input,axis=None),0.00001),
                                  true_fn=return_zero_output,
                                  false_fn=build_valid_output
                                  )


    def save_weights(self,sess,path):
        # 非常重要：这里不能够新建一个tf.Session，也不能用这个类初始化的Session，而应该用全局的sess
        v = tf.trainable_variables()
        atom_v = []
        for i in v:
            if i.name.startswith(self.atom):
                atom_v.append(i)
        #print([i for i in atom_v])

        all_vars = sess.run(atom_v)
        save_dict = {}
        for idx, _ in enumerate(all_vars):
            save_dict[atom_v[idx].name] = all_vars[idx]
        with open(path,"wb") as f:
            pickle.dump(save_dict,f)


    def load_weights(self,sess,path):
        with open(path, "rb") as f:
            save_dict = pickle.load(f)
        v = tf.trainable_variables()
        atom_v = []
        for i in v:
            if i.name.startswith(self.atom):
                atom_v.append(i)
        assign = []

        for key in save_dict:
            for v_ in v:
                if v_.name == key:
                    assign.append(tf.assign(v_,value=save_dict[key]))
        sess.run(assign)




class FullAtomModel(object):
    '''
    全原子网络，输入为None 1 的一堆某个样本某种原子的能量
    输出为None 1，为最终能量
    里面经过concat 和reduce sum
    然后调整其他训练参数
    '''
    def __init__(self,atom_cases,model_dir,feature_num):
        self.train_step = None
        self.output = None
        self.atom_cases = sorted(list(set(atom_cases)))
        self.model_dir = model_dir
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
        self.feature_num = feature_num
        self.sess = tf.Session()
        self.build_models()




    def check_atom_feature_dict(self,atom_feature_dict):
        if not isinstance(atom_feature_dict,dict):
            raise ValueError("Input must be a dict like 'H':array,'O':array")
        keys = atom_feature_dict.keys()
        if set(keys) != set(self.atom_cases):
            raise ValueError("Atom cases must be same! input: %s now: %s"
                             %(",".join(keys),",".join(self.atom_cases)))

    def predict(self,x):
        atom_feature_dict = x
        feed_dict = {}
        self.check_atom_feature_dict(atom_feature_dict)
        for i in self.input:
            feed_dict[self.input[i]] = atom_feature_dict[i]

        return self.sess.run(self.output,feed_dict=feed_dict)

    def atom_weights_path(self,atom_name):
        atoms = copy.deepcopy(self.atom_cases)
        atoms.remove(atom_name)
        return self.model_dir + "/" + atom_name +"__"+"_".join(atoms) + ".weights"

    def fit(self,x,y,epoch,batch_size=None,lr=0.0001,save_weights=True,load_weights=True):

        # 在这里判断是否所有的原子都包括在内
        string = ""
        for atom_type in x:
            string += atom_type
            string += str(x[atom_type].shape[1])
        print_file(">>Train for %s" % string)




        y = np.array(y).reshape(-1,1)
        # batch_size 还没有实现
        if self.train_step is None:
            self.build_fit_model(lr)

        if load_weights:
            try:
                self.load_atom_weights()
            except:
                pass
        atom_feature_dict = x
        feed_dict = {}
        self.check_atom_feature_dict(atom_feature_dict)
        for i in self.input:
            feed_dict[self.input[i]] = atom_feature_dict[i]
        feed_dict[self.true_y] = y
        #print(feed_dict)

        for step in range(epoch):
            self.sess.run(self.train_step,feed_dict=feed_dict)
            if step % 100 == 0:
                print_file("    Epoch    %s  " % step + "    Error:   " + str(self.sess.run(self.rmse,feed_dict=feed_dict)))
        if save_weights:
            self.save_atom_weights()
            print_file("Save Weights.")



    def fit_batch_one_step(self, x, y):
        pass


    def build_fit_model(self,lr):
        self.true_y = tf.placeholder(shape=(None,1),dtype=tf.float32)
        self.rmse = tf.sqrt(tf.reduce_mean(tf.squared_difference(self.output, self.true_y)))
        self.train_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.rmse)
        #self.train_step = tf.train.MomentumOptimizer(learning_rate=lr,momentum=0.01).minimize(self.rmse)
        self.sess.run(tf.global_variables_initializer())


    def build_models(self):
        # 临时创建新的model，实际上如果有就需要从pickle中加载
      #graph = tf.Graph()
      #with graph.as_default():
        self.models = []
        self.input = {}
        for atom in self.atom_cases:
                model = AtomModel(self.feature_num,atom=atom)
                self.models.append(model)
                self.input[atom] = model.input

        outputs = [model.output for model in self.models]
        #print(outputs)
        #print(outputs)
        concat = tf.concat(outputs, axis=1)
        #print("concat shape", concat.shape)
        self.output = tf.reduce_sum(concat, axis=1)
        #print(self.output.shape)

        self.sess.run(tf.global_variables_initializer())


      #tf.summary.FileWriter(os.getcwd() + "/my_atom_model_structure/",graph)
      #exit()


    def save_atom_weights(self):


        for atom_model in self.models:
            atom_model.save_weights(self.sess,self.atom_weights_path(atom_model.atom))

    def load_atom_weights(self):
        for atom_model in self.models:
            atom_model.load_weights(self.sess,self.atom_weights_path(atom_model.atom))
            print_file("Load Weights Success for atom %s" % atom_model.atom)

        self.debug_print_weights()

    def show_structure(self):
        with tf.Session() as sess:
            writer = tf.summary.FileWriter(os.getcwd()+"/atom_model_structure/", sess.graph)

    def debug_print_weights(self):
        print_file("Debug Print Weights>>>>>>>>>>>>>>>>")
        print_file(str(self.sess.run(tf.trainable_variables()[0]).flatten()[:10]))


def t_pred_save_load_weights():
    a = FullAtomModel(["C", "H", "O", "Ni"], os.getcwd() + "/model", 4)


    C = H = O = Ni = np.array([1,1,1,1]).reshape(1,1,4)
    dict_ = {
        "C":C,
        "H":H,
        "O":O,
        "Ni":Ni
    }

    print(a.predict(dict_))

    C = H = O  = np.array([1, 1, 1, 1,2,2,2,2,1,1,1,1,2,2,2,2]).reshape(2, 2, 4)
    Ni = np.array([1, 1, 1, 1, 2, 2, 2, 2, ]).reshape(2, 1, 4)
    dict_ = {
        "C": C,
        "H": H,
        "O": O,
        "Ni": Ni

    }

    print(a.predict(dict_))

    #tf.set_random_seed(123)

    try:
        a.debug_print_weights()
        #a.load_atom_weights()
        a.debug_print_weights()
        print("These two weighs should be different!")
    except:
        pass
    y = [0]
    a.fit(dict_,y,epoch=10)
    a.debug_print_weights()
    a.save_atom_weights()

def t_train():
    # 这里atom cases也可以为原子序数
    a = FullAtomModel(["C", "H", "O", "Ni"], os.getcwd() + "/model", 4)

    C = H = O = Ni = np.array([1, 2, 3, 4]).reshape(1, 1, 4)
    dict_ = {
        "C": C,
        "H": H,
        "O": O,
        "Ni": Ni
    }

    print(a.predict(dict_))

    y = [0.1]

    print(a.fit(dict_,y,epoch=1000,load_weights=False))

    print(a.predict(dict_))

def train_from_pkl():
    with open("C:\\Users\Administrator\Desktop\\feature.pkl", "rb") as f:
        total_train_feed_x, total_test_feed_x, total_train_feed_y, total_test_feed_y, atom_cases, n_feat = pickle.load(f)

    nn = FullAtomModel(atom_cases, os.getcwd() + "/model",n_feat)


    # 两种数据集交替训练
    for _ in range(30):
        nn.fit(total_train_feed_x[0],total_train_feed_y[0],epoch=100,load_weights=False)
        nn.fit(total_train_feed_x[1], total_train_feed_y[1], epoch=100, load_weights=False)

    pred = nn.predict(total_test_feed_x[0])

    plt.plot(pred,total_test_feed_y[0],'ro')
    plt.show()

    # 预测另一个完全不同的吸附质分子
    pred = nn.predict(total_test_feed_x[1])

    plt.plot(pred, total_test_feed_y[1], 'ro')
    plt.show()




def one_line_train():

    '''
    TODO: 数据集随机打乱训练
    TODO：增加类存储，增加数据集，所涉及原子种类信息保存等，对NN结构进行调参
    
    '''

    print_file(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>New Game Begin!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    print_file("Start Data Collecting")
    # 准备数据

    aim_vasp_path = "S:\数据集\碳纳米管掺杂\\5-5\\b\oh"

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
    # 利用DeepChem的ANI transform进行转化，存储到pkl文件中
    nn = FullAtomModel(atom_cases, os.getcwd() + "/model",n_feat)


    try:
        nn.load_atom_weights()
    except:
        print_file("Load Weights Failed")

    string = 'Total feed X shape: \n'
    string += "Train: "

    for i in total_train_feed_x:
        for j in i:
            print(i[j])
            string += j + ":" +str(i[j].shape)
    string += "\n"

    string += "Test: "
    for i in total_test_feed_x:
        for j in i:
            string += j + ":" + str(i[j].shape)
    string += "\n"
    print_file(string)

    repeat = 100
    index = 0

    for i in range(repeat):# 这里用while True也行，因为每次fit都会保存weights
        #print_file(">>Loop %s/%s"%(i+1,repeat))
        print_file(">>Loop %s"%(index))

        for dataset_index in range(len(total_train_feed_x)):
            print_file(">>>>Train for %s/%s" % (dataset_index+1, len(total_train_feed_x)))
            nn.fit(total_train_feed_x[dataset_index],total_train_feed_y[dataset_index],epoch=1000,load_weights=True)
        index += 1

    nn.save_atom_weights()

def predict():

    # 直接使用编码后的数据，也可以现进行编码
    with open("ANI_features.pkl", "rb") as f:
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




def show_weights():
    a = FullAtomModel(["C", "H", "O", "Pt"], os.getcwd() + "/model/trained", 769)
    a.load_atom_weights()
    a.debug_print_weights()




if __name__ == '__main__':
    #t_init_pred_save_load_weights()
    #t_train()
    #train_from_pkl()
    one_line_train()
    predict()

    #t_pred_save_load_weights()
    #show_weights()




