

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
 **<p align="center"> energy prediction in testset </p>**
![](https://i.imgur.com/SJDc03R.png)
 **<p align="center"> network structure </p>**
