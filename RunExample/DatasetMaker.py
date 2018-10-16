# -*- coding: utf-8 -*-


import numpy as np
import os
from VDE.VASPMoleculeFeature import VASP_DataExtract
import pickle
from DatasetOffer import print_file




class DatasetMaker(object):

    def __init__(self,dir_list):
        if not  isinstance(dir_list,list):
            dir_list = [dir_list]
        self.in_dir = dir_list
        self.vasp_dirs = []
        for i in self.in_dir:
            self.vasp_dirs.extend(self.get_vasp_dirs(i))
        print("Get total %s vasp dirs" % len(self.vasp_dirs))
        if len(self.vasp_dirs) == 0:
            raise ValueError("No vasp dirs Available")
        self.total_info = {}
        self.total_info["instance"] = "DatasetMaker"
        self.atom_cases = set([])
        for i in self.vasp_dirs:
            self.total_info[i] = {}
            self.total_info[i]["generated"] = 0
        self.b_make_dataset = 0

    def make_dataset(self):
        t = len(self.vasp_dirs)
        for i in range(t):
            print_file("Process For generating dataset: %s / %s"%(i, t))
            #print("Process for %s" % self.vasp_dirs[i])
            self.__make_one_dataset(self.vasp_dirs[i])
        self.b_make_dataset = 1


    def save_dataset(self,pkl_path):
        if self.b_make_dataset == 0:
            raise ValueError("make dataset before save dataset!")

        if os.path.isdir(pkl_path):
            pkl_path += "/atom_dataset.pkl"

        if not pkl_path.endswith(".pkl"):
            pkl_path += '.pkl'

        with open(pkl_path, "wb") as f:
            pickle.dump(self.total_info,f)

    def give_out_dataset(self):
        if self.b_make_dataset == 0:
            raise ValueError("make dataset before save dataset!")
        return self.total_info


    def __make_one_dataset(self,vasp_dir):
        test = VASP_DataExtract(vasp_dir=vasp_dir)
        test.get_atom_and_position_info()
        a = test.get_output_as_atom3Dspace()

        if len(a.atoms_pos_info) <=4: # 如果样本不够，这是很可能出现的
            print_file("No enough samples for %s, which have %s." % (vasp_dir, len(a.atoms_pos_info)))
            del self.total_info[vasp_dir]
            return

        print_file("vasp_dir %s have sample %s" % (vasp_dir, len(a.atoms_pos_info)))

        self.total_info[vasp_dir]["generated"] = 1
        # 这里的x y不是坐标而是坐标x和能量y
        self.total_info[vasp_dir]['x'], self.total_info[vasp_dir]['y'], atom_cases = a.generate_data()
        self.atom_cases = self.atom_cases.union(atom_cases)
        print("AtomCases",self.atom_cases)

        self.total_info[vasp_dir]['atom_cases'] = self.atom_cases

        # 这里要以一系列数据集为核心建立模型，包含所有的原子




    def get_vasp_dirs(self,dir):

            files = os.walk(dir)
            vasp_dir = []
            for i in files:
                if "OUTCAR" in i[2]:
                    vasp_dir.append(i[0])
            return vasp_dir


if __name__ == '__main__':
    aim_vasp_path = "C:\\Users\wang\Desktop\运行结果\嵌套运行结果\interm\Pt\AllSurfaceG1"
    temp = DatasetMaker(aim_vasp_path)
    temp.make_dataset()
    print(temp.total_info)
    #temp.save_dataset("C:\\Users\wang\Desktop\运行结果")
# 现在的问题是，OUT.ANI有2倍的坐标数据于能量数据： 最终选择：直接从OUTCAR中提取坐标，寻找以前有的代码

# TODO  数据集需要规定原子种类，如果有没有的原子，它的结果不应该增加到能量中，因为有bias的存在，所以不能用feature为0000来进行