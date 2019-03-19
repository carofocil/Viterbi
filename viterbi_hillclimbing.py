import time
import array
import pickle
import codecs
import decimal
import numpy as np
import random as r
from random import shuffle
from random import randint, random


""" This program use pickle for deserialize states_tuples, start_prob, e_pro, transition
Also, applies Differential evolution for optimizacion of HMM parameters through Viterbi algorithm using conventional form lineal.
The values of A, B, C are random lists
STEP 2"""

class hmm_ga():
    s = codecs.open(".//states_tuples.pkl", "rb")
    states_tuples = pickle.load(s)
    s.close()

    sp = codecs.open(".//start_prob.pkl", "rb")
    start_prob = pickle.load(sp)
    sp.close()

    e = codecs.open(".//epro.pkl", "rb")
    e_pro = pickle.load(e)
    e.close()

    t = codecs.open(".//trans_p.pkl", "rb")
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
            sumarf, sumac = 0, 0
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

        data_f1 = 0
        class_names = self.m_confusion.keys()
        for i in range(len(class_names)):
            data_f1 += f1[i]
        data_f1 = data_f1 / 5

        return data_f1

    def viterbi(self, obs, classes_test, transition, emission):
        V = [{}]
        for st in self.states_tuples:
            V[0][st] = {"prob": self.start_prob[st] * reduce(lambda x, y: x * y, [emission[st][obs[0][feat]] if emission[st].has_key(obs[0][feat]) else 0.000001 for feat in range(len(obs[0]))]), "prev": None}

        for t in range(1, len(obs)):
            V.append({})
            for st in self.states_tuples:
                max_tr_prob = max(V[t - 1][prev_st]["prob"] * transition[prev_st][st] for prev_st in self.states_tuples)
                for prev_st in self.states_tuples:
                    if V[t - 1][prev_st]["prob"] * transition[prev_st][st] == max_tr_prob:
                        max_prob = max_tr_prob * reduce(lambda x, y: x * y, [emission[st][obs[t][feat]] if emission[st].has_key(obs[t][feat]) else 0.000001 for feat in range(len(obs[t]))])
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

    def test_phase(self, txt_test, classes_test, transition, emission):
        self.matrix_confusion()
        error = 0.0
        total = 0
        correct = 0
        i = 0
        for obs_space in range(len(txt_test)):
            predicted_class = self.viterbi(txt_test[obs_space], classes_test[obs_space], transition, emission)
            total += len(txt_test[obs_space])
            for element in range(len(predicted_class)):
                if predicted_class[element] == classes_test[obs_space][element]:
                    correct += 1
                self.m_confusion[classes_test[obs_space][element]][predicted_class[element]] += 1

        performance = correct / float(total)
        error = 1 - performance
        f_m = self.recall_precision()
        return performance, error, f_m

    def convert_matrix(self, ind):
        """ This converts a list into transition matrix and emission probability """
        transition = {}
        emission = {}
        i = 0
        j = 0
        for st1 in self.trans_p.keys():
            transition[st1] = {}
            for st2 in self.trans_p.keys():
                transition[st1][st2] = ind[i]
                i += 1
        j = i
        for st in self.e_pro.keys():
            emission[st] = {}
            for word in self.e_pro[st].keys():
                emission[st][word] = ind[j]
                j += 1
        return transition, emission

    def genetic_algorithm(self, txt_test, classes_test):
        # Join e_pro and trans_p due to genetic algorithm receives a list, nor a dictionary
        m_transition = [self.trans_p[key][key2] for key in self.trans_p.keys() for key2 in self.trans_p[key] ]
        m_emission = [self.e_pro[state][word] for state in self.e_pro.keys() for word in self.e_pro[state]]
        ga = m_transition + m_emission

        v_v = np.copy(ga)
        v = r.sample(xrange(len(ga)), len(ga))
        #grad = round(abs(random()), 2)
        grad = 0.8


        v_v_transition, v_v_emission = self.convert_matrix(v_v)  # For create matrix transition and emission probability
        v_v_fitness, v_v_error, v_v_recall = self.test_phase(txt_test, classes_test, v_v_transition, v_v_emission)
        print "original", v_v_error, v_v_fitness, v_v_recall

        a = 0
        for i in v[:1000]:
            v_v_transition, v_v_emission = self.convert_matrix(v_v)  # For create matrix transition and emission probability
            v_v_fitness, v_v_error, v_v_recall = self.test_phase(txt_test, classes_test, v_v_transition, v_v_emission)
            print len(v_v)- a, i, "original", v_v_error, v_v_fitness, v_v_recall

            v_v_p = np.copy(v_v)
            v_v_p[i] = v_v[i] + grad
            v_v_p_transition, v_v_p_emission = self.convert_matrix(v_v_p)  # For create matrix transition and emission probability
            v_v_p_fitness, v_v_p_error, v_v_p_recall = self.test_phase(txt_test, classes_test, v_v_p_transition, v_v_p_emission)
            print len(v_v)- a, i, "suma", v_v_p_error

            v_v_s = np.copy(v_v)
            v_v_s[i] = v_v[i] - grad
            v_v_s_transition, v_v_s_emission = self.convert_matrix(v_v_s)  # For create matrix transition and emission probability
            v_v_s_fitness, v_v_s_error, v_v_s_recall = self.test_phase(txt_test, classes_test, v_v_s_transition, v_v_s_emission)
            print len(v_v)- a, i, "resta", v_v_s_error

            if v_v_p_error < v_v_error and v_v_p_error < v_v_s_error:
                flag = v_v_p_error + grad

                while v_v_p_error < flag:
                    v_v_p_fitness, flag, v_v_p_recall = self.test_phase(txt_test, classes_test, v_v_p_transition, v_v_p_emission)
                    v_v_p[i] = v_v_p[i] + grad
                    v_v_p_transition, v_v_p_emission = self.convert_matrix(v_v_p)
                    v_v_p_fitness, v_v_p_error, v_v_p_recall = self.test_phase(txt_test, classes_test, v_v_p_transition, v_v_p_emission)
                    print len(v_v)- a, i, "mejora VVP", v_v_p_error

            if v_v_s_error < v_v_error and v_v_s_error < v_v_p_error:
                flag = v_v_s_error + grad

                while v_v_s_error < flag:
                    v_v_s_fitness, flag, v_v_s_recall = self.test_phase(txt_test, classes_test, v_v_s_transition, v_v_s_emission)
                    v_v_s[i] = v_v_s[i] - 5
                    v_v_s_transition, v_v_s_emission = self.convert_matrix(v_v_s)
                    v_v_s_fitness, v_v_s_error, v_v_s_recall = self.test_phase(txt_test, classes_test, v_v_s_transition, v_v_s_emission)
                    print len(v_v)- a, i, "Mejora en VVS:", v_v_s_error

            if v_v_s_error < v_v_p_error and v_v_s_error < v_v_error:
                v_v = np.copy(v_v_s)
            elif v_v_p_error < v_v_s_error and v_v_p_error < v_v_error:
                v_v = np.copy(v_v_p)
            elif v_v_error < v_v_p_error and v_v_error < v_v_s_error:
                v_v = np.copy(v_v)
            a += 1
            print "--------------------------------------------------------------------\n"

        return v_v

    def main(self):
        data_train = codecs.open(".//train+dev.txt", "rb", "utf-8") #lineal

        "TRAIN INFORMATION"
        b = []
        c = []
        txt_train = []
        txt_test = []
        classes_test = []
        for line in data_train:
            line = line.lower()
            if not line.startswith("\n"):
                b.append(list(line.split(" ")[:-2]))
                c.append(line.split(" ")[-2])
            else:
                txt_test.append(b)
                classes_test.append(c)
                b = []
                c = []
        txt_test.append(b)
        classes_test.append(c)

        fvector = self.genetic_algorithm(txt_test, classes_test)  # Gives  the best value of d according to Viterbi precision

        v_transition, v_emission = self.convert_matrix(fvector)  # For create matrix transition and emission probability
        v_fitness, v_error, v_recall = self.test_phase(txt_test, classes_test, v_transition, v_emission)

        print "Final", v_fitness, v_error, v_recall
        e_file = open(".//bestvector.pkl", "wb")
        pickle.dump(fvector, e_file)
        e_file.close()

hmm = hmm_ga()
hmm.main()