#!/usr/bin/env python3

import numpy as np
import os
import sys
import json

import wfdb
from tensorflow import keras
from tqdm import tqdm
from utils import qrs_detect, comp_cosEn, save_dict

"""
Written by:  Xingyuan Ying, Qing Pan, Ziyou Zhang(CPSC_eie)
             School of Information Engineering
             Zhejiang University or Technology, China
             2026987930@qq.com

tensorflow = 2.2.0

Save answers to '.json' files, the format is as {‘predict_endpoints’: [[s0, e0], [s1, e1], …, [sm-1, em-2]]}.
"""

def load_data(sample_path):
    sig, fields = wfdb.rdsamp(sample_path,channels=[0])
    length = len(sig)
    fs = fields['fs']

    return sig, length, fs

def ngrams_rr(data, length):
    grams = []
    for i in range(0, length-12, 12):
        grams.append(data[i: i+12])
    return grams

def challenge_entry(sample_path):
    """
    This is a baseline method.
    """

    sig, _, fs = load_data(sample_path)
    sig1=sig
    end_points = []

    sample = wfdb.rdann(sample_path,'atr').sample
    L = 160
    ldict = { 0:'(AFIB', 1:'(N'}#0:'(AFL',
    
    model = keras.models.load_model('best_weights.h5')
    result = []
    for i in tqdm(range(sample.shape[0] - 1)):
        x = sig[sample[i]:sample[i + 1]]
        x = np.concatenate([x,np.zeros((L,1))])
        x = x[:L]
        result.append(ldict[model.predict(np.array([x])).argmax()])
    
    #print(result)
    count = 0
    #count1 = 0
    length=len(result)
    
    
    def countStr(result0 , begin):
        cnt = 1
        for t in range(5):
            if result0[begin] != result0[begin + 1]:
                return cnt
            elif cnt == 5:
                return cnt
            begin = begin + 1
            cnt = cnt + 1
    
    for j in range(0,length - 1):
        s = result[j] == result [j + 1]
        if s == False and j< length-7:
            count = countStr(result, j+1)
            #count1 = countStr(result, j+2)
            if count == 1 :
                result[j+1] = result[j]
    
    
    k = [i for i, x in enumerate(result) if x=='(AFIB']
    
    if len(k) !=0: 
        
        def split_num_l(num_lst):
            """
            """
            num_lst_tmp = [int(n) for n in num_lst]
            sort_lst = sorted(num_lst_tmp)  # ascending
            len_lst = len(sort_lst)
            i = 0
            split_lst = []
            
            tmp_lst = [sort_lst[i]]
            while True:
                if i + 1 == len_lst:
                    break
                next_n = sort_lst[i+1]
                if sort_lst[i] + 1 == next_n:
                    tmp_lst.append(next_n)
                else:
                    split_lst.append(tmp_lst)
                    tmp_lst = [next_n]
                i += 1
            split_lst.append(tmp_lst)
            return split_lst
        
        
        def mg_str_lst(mylst):
            """
            """
            b = 0
            c = 0
            fsum = 0
            rst = 0
            mg_l = []
            ms_l = []
            l = len(sample)
            for num_l in mylst:
                if len(num_l) > 2:
                    b = num_l[0]
                    c = num_l[-1]
                    if b == 0:
                    #if c-b>=5:
                        fsum = fsum + c - b + 1
                        rst = 1
                        bs = float(sample[b])
                        cs = float(sample[c])
                        ms_l.append(bs)
                        ms_l.append(cs)
                        mg_l.append(ms_l)
                        ms_l = []
                    elif rst == 1 and c >= l -2:
                        fsum = fsum + c - b + 1
                        if fsum >= 0.85*(l-1):
                            cs = float(sample[c])
                            ms_l.append(0.0)
                            ms_l.append(cs)
                            mg_l = ms_l
                            ms_l = []
                        else:
                            bs = float(sample[b])
                            cs = float(sample[c])
                            ms_l.append(bs)
                            ms_l.append(cs)
                            mg_l.append(ms_l)
                            ms_l = []
                    else:
                        fsum = fsum + c - b + 1
                        bs = float(sample[b])
                        cs = float(sample[c])
                        ms_l.append(bs)
                        ms_l.append(cs)
                        mg_l.append(ms_l)
                        ms_l = []
            return mg_l
    
        in_lst = k
        lst_split = split_num_l(in_lst)
        lst_mg = mg_str_lst(lst_split)
        #print(lst_mg)
    
    else:
        lst_mg = k
    
    json_lst=json.dumps(lst_mg)
    pred_dcit = {'predict_endpoints': json_lst}
    
    return pred_dcit


if __name__ == '__main__':
    DATA_PATH = sys.argv[1]
    RESULT_PATH = sys.argv[2]
    #DATA_PATH = r"D:\wfdb\data\new\training_I" 
    #RESULT_PATH = r"D:\wfdb\json1" 
    if not os.path.exists(RESULT_PATH):
        os.makedirs(RESULT_PATH)
        
    test_set = open(os.path.join(DATA_PATH, 'RECORDS'), 'r').read().splitlines()
    for i, sample in enumerate(test_set):
        print(sample)
        sample_path = os.path.join(DATA_PATH, sample)
        pred_dict = challenge_entry(sample_path)

        save_dict(os.path.join(RESULT_PATH, sample+'.json'), pred_dict)

