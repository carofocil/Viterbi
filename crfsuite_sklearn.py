import codecs
import pickle
import sklearn
import scipy.stats
from sklearn.metrics import make_scorer
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import RandomizedSearchCV

import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics

def word2features(sent, i):
    word = sent[i][0]
    no_feat = 21
    features = ({'bias': 1.0, 'word': word})
    for j in range(1,no_feat):
        if not "word_features[" + str(j) +  "]"  in features:
            features["word_features[" + str(j) +  "]"] = sent[i][j]
        
    if i > 0:
        word1 = sent[i-1][0]
        features.update({'-1:word': word1})
        for j in range(1, no_feat):
            if not "-1:word_features[" + str(j) +  "]" in features:
                features["-1:word_features[" + str(j) +  "]"] = sent[i-1][j]           
    else:
        features['BOS'] = True
         
    if i < len(sent)-1:
        word_plus_1 = sent[i+1][0]
        features.update({'+1:word': word_plus_1 })
        features_plus_1 = { } 
        for j in range(1, no_feat):
            if not "+1:word_features[" + str(j) +  "]" in features:
                features["+1:word_features[" + str(j) +  "]" ] = sent[i+1][j]          
    else:
        features['EOS'] = True
    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [pattern[-1] for pattern in sent]


def readtxt(txt_train):
    b = []
    train_sents = []
    for line in txt_train:
        if not line.startswith("\n"):                          
            b.append(line.split())
        else:
            train_sents.append(b)
            b = []
    train_sents.append(b)
    return train_sents
   
def main():
    print ("Here")
    txt_train = codecs.open(".//train.txt", "rb", encoding = "utf-8")
    txt_test = codecs.open(".//test.txt", "rb", encoding = "utf-8")
    
    train_sents = readtxt(txt_train)
    test_sents = readtxt(txt_test)
    
      
    X_train = [sent2features(s) for s in train_sents]
    y_train = [sent2labels(s) for s in train_sents]

    X_test = [sent2features(s) for s in test_sents]
    y_test = [sent2labels(s) for s in test_sents]
    

    #STEP3: Training
    crf = sklearn_crfsuite.CRF(
        algorithm = 'lbfgs', 
        c1 = 0.1, 
        c2 = 0.1, 
        max_iterations = 100, 
        all_possible_transitions = True
    )
    crf.fit(X_train, y_train)

    #STEP4: Evaluation
    labels = list(crf.classes_)
    print ("it's here")
    y_pred = crf.predict(X_test)
    
    y_file = open(".//Task1.pkl", "wb")
    pickle.dump(y_pred, y_file)
    y_file.close()
        
    metrics.flat_f1_score(y_test, y_pred,  average='weighted', labels=labels)
       
    sorted_labels = sorted(labels, key=lambda name: (name[1:], name[0]))
    
    #Inspect per-class results in more detail:
    print(metrics.flat_classification_report(y_test, y_pred, labels=sorted_labels, digits=3))
   
    #Accuracy
    print (len(y_test), len(y_pred))
    t = 0
    total = 0
    for i in range(len(y_test)):
        for j in range(len(y_test[i])):
            if y_test[i][j] == y_pred[i][j]:
                t += 1
                total += 1
            else:
                total += 1

    print (t, total, t/total) 

if __name__ == "__main__":
    main()