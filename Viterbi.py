
# coding: utf-8

# In[ ]:




# In[5]:




# In[2]:

import time
import array
import pickle
import codecs
import decimal
import numpy as np
import matplotlib.pyplot as plt
import functools
from random import shuffle
from random import random, randint

""" This program use pickle for deserialize states_tuples, start_prob, e_pro, transition
Also, applies Differential evolution for optimizacion of HMM parameters through Viterbi algorithm using conventional form lineal.
The values of A, B, C are random lists
STEP 2"""

class hmm_ga():
    """BASED ON THE TASK, YOU MUST CHANGE THIS PART"""
    s = codecs.open(".\\states.pkl", "rb")
    states_tuples = pickle.load(s)
    s.close()

    sp = codecs.open(".\\start_prob.pkl", "rb")
    start_prob = pickle.load(sp)
    sp.close()

    e = codecs.open(".\\epro.pkl", "rb")
    e_pro = pickle.load(e)
    e.close()

    t = codecs.open(".\\transition.pkl", "rb")
    trans_p = pickle.load(t)
    t.close()
    
    m_confusion = {}

    def matrix_confusion(self):
        for i in self.states_tuples:
            m = {}
            for j in self.states_tuples:
                m[j] = 0
            self.m_confusion[i] = m

    def recall_precision(self):
        recall = []
        precision = []
        f1 = []
        diag = []
        diag_inv = []
        suma_row = []
        suma_col = []
        FP = []
        #print "\nMATRIX CONFUSION\n"
        for i in self.m_confusion.keys():
            sumarf = 0
            sumac = 0
            print (i, self.m_confusion[i])
            for j in self.m_confusion.keys():
                sumarf += self.m_confusion[i][j]
                sumac += self.m_confusion[j][i]
                if i == j:
                    diag.append(self.m_confusion[i][j])
                else:
                    diag_inv.append(self.m_confusion[i][j])
            suma_row.append(sumarf)
            suma_col.append(sumac)

        # Calculate recall and precision
        r = 0
        p = 0
        t_n = 0
        for i in range(len(diag)):
            r = diag[i] / float(suma_row[i])
            p = diag[i] / float(suma_col[i])
            t_n = diag_inv[i] / float(suma_row[i])
            recall.append(r)
            precision.append(p)
            FP.append(t_n)

        fm = 0
        for i in range(len(precision)):
            fm = (2 * precision[i] * recall[i]) / float(precision[i] + recall[i])
            f1.append(fm)

        data_recall = 0
        f_measure = 0
        class_names = self.m_confusion.keys()
        print ("\n\nClase ", "\tPrecision", "\tRecall", "\t\tF-measure")
        for cl in range(len(class_names)):
            print (list(class_names)[cl], "\t", format(precision[cl], '.4f'), "\t", format(recall[cl], '.4f'), "\t", format(f1[cl], '.4f'))
            f_measure += f1[cl] 
        return (f_measure/len(class_names))

    def viterbi(self, obs, classes_test):
#         try:
        V = [{}]
        for st in self.states_tuples:
            V[0][st] = {"prob": self.start_prob[st] * functools.reduce(lambda x, y: x * y, [self.e_pro[st][feat][obs[0][feat]] if (obs[0][feat]) in self.e_pro[st][feat] else 0.00001 for feat in range(len(obs[0]))]), "prev": None}

        for t in range(1, len(obs)):
            V.append({})
            for st in self.states_tuples:
                max_tr_prob = max(V[t - 1][prev_st]["prob"] * self.trans_p[prev_st][st] for prev_st in self.states_tuples)
                for prev_st in self.states_tuples:
                    if V[t - 1][prev_st]["prob"] * self.trans_p[prev_st][st] == max_tr_prob:
                        max_prob = max_tr_prob * functools.reduce(lambda x, y: x * y, [self.e_pro[st][feat][obs[t][feat]] if (obs[t][feat]) in self.e_pro[st][feat] else 0.00001 for feat in range(len(obs[t]))])
                        V[t][st] = {"prob": max_prob, "prev": prev_st}
                        break
        opt = []
        max_prob = max(value["prob"] for value in V[-1].values())
        previous = None

        a = []
        for st, data in V[-1].items():
            if data["prob"] == max_prob and max_prob != 0.0:
                opt.append(st)
                previous = st
                break
            elif max_prob == 0.0:
                a.append(st)
        if a != []:
            opt.append(a[len(a)-1])
            previous = a[len(a)-1]

        for t in range(len(V) - 2, -1, -1):
            opt.insert(0, V[t + 1][previous]["prev"])
            previous = V[t + 1][previous]["prev"]

        return opt
#         except:  
#             print (len(obs), obs)
#             return ['NO' for i in obs]

    def test_phase(self, txt_test, classes_test):
        self.matrix_confusion()
        total = 0
        correct = 0
        i = 0
        f_measure = 0
        for obs_space in range(len(txt_test)):
            predicted_class = self.viterbi(txt_test[obs_space], classes_test[obs_space])
            total += len(txt_test[obs_space])
            for element in range(len(predicted_class)):
                if predicted_class[element] == classes_test[obs_space][element]:
                    correct += 1
                self.m_confusion[classes_test[obs_space][element]][predicted_class[element]] += 1
        performance = correct / float(total)
        error = 1 - performance
        f_measure = self.recall_precision()
        return performance, error, f_measure


    def main(self):
        "TEST INFORMATION"
        data_test = codecs.open(".\\test.txt", "rb", "utf-8")
           
        "TEST INFORMATION"
        b = []
        c = []
        txt_test = []
        classes_test = []
        for line in data_test:
            if not line.startswith("\n"):                           #TASK1
                b.append(list(line.split(" ")[:-1]))
                c.append(line.rstrip("\n").split(" ")[-1])

            else:
                txt_test.append(b)
                classes_test.append(c)
                b = []
                c = []
        txt_test.append(b)
        classes_test.append(c)
    
        performance, error, f1 = self.test_phase(txt_test, classes_test)
        print ("\n\nThe performance is: ", round(performance,4))
        print ("F_measure is: ", round(f1,4))

hmm = hmm_ga()
hmm.main()


# In[ ]:



