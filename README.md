

### function
- Give the dir of your VASP result dirs, the program will transfer them to dataset automatically, then use ANI transform from DeepChem to get features.
- The freatures will go into a Dense NN to predict energy, during the train and test process, the features are feed into different NN according to their atom index, and finally reduce_sum to get total energy.
- When you predict other datasets, **No Limitation to Atom Numbers**(since the final layer is reduce_sum),but **the Atom Cases Must Included in the Trainset**. E.g., if your trainset use C2H4O1Cu20, you can predict CHO, C1Cu20, O1Cu20, ..., but you can predict CHNCu. (But **Make Sure the Distribution of Your Trainset is Large Enough for Prediction**).

### install
- install VDE to get dataset from VASP result dir, [https://github.com/B-C-WANG/VDE-VaspDataExtract](https://github.com/B-C-WANG/VDE-VaspDataExtract "VaspDataExtract")
- install tensorflow, numpy, ...
- install My Version of DeepChem, no RDKit need, worked well on Windows @ [https://github.com/B-C-WANG/deepchem](https://github.com/B-C-WANG/deepchem). Run setup.py install or just copy to site-packages (it works for me to run it on HPC)

### how to run
- Set your VASP dir in AtomModel.py - one\_line\_train()
- ![](https://i.imgur.com/jOrKYtT.png)
- then run it.
- Parse the code or wait for more details in the future.

### some figs
![](https://i.imgur.com/iYZN0mu.png)
 **<p align="center"> energy prediction in testset, use transition state and intermediates on Ni surface</p>**
![](https://i.imgur.com/SJDc03R.png)
 **<p align="center"> network structure </p>**

### 算法
- 给定输入坐标
- 使用DeepChem的ANITransform得到feature，将feature转化为每个原子为中心的feature，feature的向量长度就是之后NN的输入shape的最后一维
- 然后将feature按照原子归类，转为字典，key是原子序数，value是(None, n_feature)大小的矩阵，每个样本都这样进行转化
- 网络的input是一个字典，key是原子序数，value是tf.placeHolder，这样可以实现对于不同种类原子使用不同的NN，然后将输入(None, n_feature)经过Dense得到(None, 1),reduce_sum得到1，即这种原子的能量分量，同样得到其他原子的分量，进行第二次reduce_sum，得到总能量

### 代码
- 获取某个文件夹下所有带OUTCAR文件的作为目标文件夹，以路径作为Key
- 使用我的VDE模块提取出每一步的能量和相应的坐标
- 这里把一个Vasp文件夹称作样本，里面每一步的能量和坐标称为帧
- 每一帧都通过ANI进行转化，得到每一帧按照原子分类的转化后的Feature
- 将Feature输入NN中进行训练，得到权重存储
- 进行预测的样本也需要进行ANI转化，NN读取权重进行预测