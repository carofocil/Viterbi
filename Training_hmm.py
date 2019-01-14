
# coding: utf-8

# In[8]:


import codecs
import time
import array
import numpy as np
import pickle
import decimal
import functools
from random import random
from random import shuffle


class hmm_ga():
    prev_tag = "start_hmm"
    tran_prob = {}  # This represents {'fever|healthy': 0.3, 'healthy|fever': 0.4, 'healthy|healthy': 0.7, 'fever|fever': 0.6}
    trans_p = {}  # This represents {'healthy': {'healthy': 0.7, 'fever': 0.3}, 'fever': {'healthy': 0.4, 'fever': 0.6}}

    e_pro = {}  # This representes {'normal|healthy': 0.5, 'dizzy|healthy': 0.1, 'dizzy|fever': 0.6, 'cold|healthy': 0.4, 'cold|fever': 0.3, 'normal|fever': 0.1}
    states = {prev_tag: 1}
    states_tuples = []  # This has states in list, then the list is transformed into a tuple

    observations = {}  # sequence of obervations
    start_prob = {}
    frequency = {}
    m_confusion = {}

    def transition_probability(self):
        """ Compute the transition matrix"""
        for key1 in self.states.keys():
            b = {}
            for key2 in self.states.keys():
                b[key2] = 0
            self.trans_p[key1] = b

        for i in self.trans_p.keys():
            total = 0
            for j in self.trans_p.keys():
                if (j + "|" + i) in self.tran_prob:
                    total += self.tran_prob[j + "|" + i]
                    self.trans_p[i][j] = self.tran_prob[j + "|" + i]
            for k in self.trans_p.keys():
                self.trans_p[i][k] = self.trans_p[i][k] / float(total)

    def initial_probability(self, total):
        """ Compute the start probability of each state """
        for key in self.states.keys():
            self.states_tuples.append(key)
            self.start_prob[key] = self.states[key] / float(total)
        self.states_tuples = tuple(self.states_tuples)

    def probabilities(self):
        """ Compute the probability of a word given a state and then self.frequency is  used by def emission_probability """
        for key in self.frequency.keys():
            k = key.split("|")[2]
            if k in self.states:
                self.frequency[key] = self.frequency[key] / float(self.states[k])

        for ele in self.frequency:
            classes = ele.split("|")[-1]
            num_feat = int(ele.split("|")[0])
            word = ele.split("|")[1]
            if not classes in self.e_pro:
                self.e_pro[classes] = {}
                if not num_feat in self.e_pro[classes]:
                    self.e_pro[classes][num_feat] = {}
                    if not word in self.e_pro[classes][num_feat]:
                        self.e_pro[classes][num_feat][word] = self.frequency[ele]
            else:
                if not num_feat in self.e_pro[classes]:
                    self.e_pro[classes][num_feat] = {}
                    if not word in self.e_pro[classes][num_feat]:
                        self.e_pro[classes][num_feat][word] = self.frequency[ele]
                else:
                    if not word in self.e_pro[classes][num_feat]:
                        self.e_pro[classes][num_feat][word] = self.frequency[ele]
                    else:
                        self.e_pro[classes][num_feat][word] = self.frequency[ele]

    def main(self):
        data_train = codecs.open(".\\train.txt", "rb", "utf-8")

        "TRAIN INFORMATION"
        txt_train = []
        a = []
        for line in data_train:
            if not line.startswith("\n"):                         #TASK2
                a.append(tuple(line.rstrip().split(" ")))
            else:
                txt_train.append(a)
                a = []
        txt_train.append(a)

        "TRAINING PHASE"
        total_states = 0
        for sent in txt_train:
            prev_tag = "start_hmm"
            for token in sent:
                tag = token[-1]
                if tag in self.states:
                    self.states[tag] += 1
                    total_states += 1
                else:
                    self.states.update({tag: 1})
                    total_states += 1
                if (tag + '|' + prev_tag) in self.tran_prob:
                    self.tran_prob[tag + '|' + prev_tag] += 1
                else:
                    self.tran_prob.update({tag + '|' + prev_tag: 1})
                prev_tag = tag

        del self.states['start_hmm']

        #This learn the features which are in each word. For example, frequency of each feature by class.
        for sent in txt_train:
            for token in sent:
                tag = token[-1]
                for feat in range(len(token)-1):
                    if (str(feat) + '|' + token[feat] + '|' + tag) in self.frequency:
                        self.frequency[str(feat) + '|'+ token[feat] + '|' + tag] += 1
                    else:
                        self.frequency.update({str(feat) + '|'+ token[feat] + '|' + tag : 1})

        self.probabilities()
        self.initial_probability(total_states)
        self.transition_probability()

        transition_file = open(".\\transition.pkl", "wb")
        pickle.dump(self.trans_p, transition_file)
        transition_file.close()

        states_file = open(".\\states.pkl", "wb")
        pickle.dump(self.states_tuples, states_file)
        states_file.close()

        probinitial_file = open(".\\start_prob.pkl", "wb")
        pickle.dump(self.start_prob, probinitial_file)
        probinitial_file.close()

        epro_file = open(".\\epro.pkl", "wb")
        pickle.dump(self.e_pro, epro_file)
        epro_file.close()
        
        print ("FIN")

hmm = hmm_ga()
hmm.main()


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



