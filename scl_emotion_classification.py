"""
Structural Correspondence Learning (SCL) implementation for emotion classification task proposed by:

@inproceedings{Blitzer07Biographies,
  author = {John Blitzer and Mark Dredze and Fernando Pereira},
  title = {Biographies, Bollywood, Boom-boxes and Blenders: Domain Adaptation for Sentiment Classification},
  booktitle = "Association for Computational Linguistics",
  address = "Prague, Czech Republic",
  year = "2007"
}


The data sets used for this project are from Unify-emotion-datasets created by:

@inproceedings{Bostan2018,
  author = {Bostan, Laura Ana Maria and Klinger, Roman},
  title = {An Analysis of Annotated Corpora for Emotion Classification in Text},
  booktitle = {Proceedings of the 27th International Conference on Computational Linguistics},
  year = {2018},
  publisher = {Association for Computational Linguistics},
  pages = {2104--2119},
  location = {Santa Fe, New Mexico, USA},
  url = {http://aclweb.org/anthology/C18-1179},
  pdf = {http://aclweb.org/anthology/C18-1179.pdf}
}


The implementation of SCL are based on:
https://github.com/yftah89/structural-correspondence-learning-SCL

@InProceedings{ziser-reichart:2017:CoNLL,
  author    = {Ziser, Yftah  and  Reichart, Roi},
  title     = {Neural Structural Correspondence Learning for Domain Adaptation},
  booktitle = {Proceedings of the 21st Conference on Computational Natural Language Learning (CoNLL 2017)},
  year      = {2017},  
  pages     = {400--410},
}

The implementation of data extraction are based on:
https://github.com/sarnthil/unify-emotion-datasets/

@inproceedings{Bostan2018,
  author = {Bostan, Laura Ana Maria and Klinger, Roman},
  title = {An Analysis of Annotated Corpora for Emotion Classification in Text},
  booktitle = {Proceedings of the 27th International Conference on Computational Linguistics},
  year = {2018},
  publisher = {Association for Computational Linguistics},
  pages = {2104--2119},
  location = {Santa Fe, New Mexico, USA},
  url = {http://aclweb.org/anthology/C18-1179},
  pdf = {http://aclweb.org/anthology/C18-1179.pdf}
}


"""


from __future__ import print_function
import json
import collections
import random
import numpy as np
import pickle

import regex as re
import glob

from datetime import datetime

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mutual_info_score
from sklearn.decomposition import TruncatedSVD

from sklearn import linear_model
from sklearn.linear_model import LogisticRegression


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

from sklearn.metrics import hinge_loss

from sklearn.naive_bayes import MultinomialNB

#import lime
import sklearn
import sklearn.ensemble
import sklearn.metrics
#from lime import lime_text
from sklearn.pipeline import make_pipeline
#from lime.lime_text import LimeTextExplainer




# This function extracts the data for given domain from unified-dataset.jsonl
# The implementation of the function is based on: 
# https://github.com/sarnthil/unify-emotion-datasets/blob/master/classify_xvsy_logreg.py
def get_raw_data(domain_name):
    ekmans_labels = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
    text_instances, labels = [], []
    with open("data/unified-dataset.jsonl") as f:
        for line in f:
            data = json.loads(line)
            # extract texts and labels
            if data["source"] == domain_name:
                # extract emotional label in scr
                for emotion in data["emotions"]:
                    if data["emotions"][emotion] == 1 and emotion in ekmans_labels:
                        text_instances.append(data["text"])
                        labels.append(emotion)
                    #for affectivetext
                    if type(data["emotions"][emotion]) == float and data["emotions"][emotion] > 0.5 and emotion in ekmans_labels:
                        text_instances.append(data["text"])
                        labels.append(emotion)
    
    counter=collections.Counter(labels)
    #print("Labels distribution for", domain_name.upper(), "extracted from unify-emotion-datasets:")
    #print(counter, "\n")
    return text_instances, labels





# This function extracts the data for training (500 samples per label), testing(113 samples per label) and unlabeled (the rest)
def split_data_balanced(data, data_labels):
    train_data = []
    train_labels = []
    
    test_data = []
    test_labels = []
    
    unl_data = []
    
    # extract unlabeled and 500 intances per label
    for i in range(len(data_labels)):
        if data_labels[i] in train_labels:
            # Count how often does each label occurs in the train set
            counter=collections.Counter(train_labels)
            if counter[data_labels[i]] < 500: 
                train_labels.append(data_labels[i])
                train_data.append(data[i])
            elif data_labels[i] in test_labels:
                # Count how often does each label occurs in the test set
                counter=collections.Counter(test_labels)
                if counter[data_labels[i]] < 113: 
                    test_labels.append(data_labels[i])
                    test_data.append(data[i])
                else:
                    # remaining data as unlabelled
                    unl_data.append(data[i])
            else:
                test_labels.append(data_labels[i])
                test_data.append(data[i])
                
        else: 
            train_labels.append(data_labels[i])
            train_data.append(data[i])
   
    
    
    counter=collections.Counter(train_labels)
    #print("Instances distribution in SRC based on Ekman's model with max. 500 examples per label for TRAINING:")
    #print(counter, "\n")
    counter=collections.Counter(test_labels)
    #print("Instances distribution in TRG based on Ekman's model with max. 113 examples per label for TESTING:")
    #print(counter, "\n")
    
    
    return train_data, train_labels, test_data, test_labels, unl_data


def evaluate(true, pred):
    if len(true) != len(pred):
        print("Enter correct inputs for evaluation! \n")
    #data types for model introspection
    domain_statistics = {}
    labels = np.unique(true)
    for i in range(len(labels)):
        tp_indexes, fp_indexes, tn_indexes, fn_indexes = [], [], [], []
        label = labels[i]
        tp = 0 
        fp = 0 
        tn = 0
        fn = 0
        precision = 0.0
        recall = 0.0
        acc = 0.0
        f1 = 0.0
        #prüge, ob es mit 0 anfängt!!!
        #for i in range(len(true)): 
        for i, item in enumerate(true, start=0):
            #prüfe die Gleichheit zweier str aus der Liste, weil
            #-fps sind manchmal falsch ausgewertet
            if true[i] == pred[i] == label:
                tp +=1
                tp_indexes.append(i)
            if true[i] != pred[i] and pred[i] == label: # label != true[i]
                fp +=1 
                fp_indexes.append(i)
            if true[i] == pred[i] != label:
                tn +=1
                tn_indexes.append(i)
            if true[i] != pred[i] and true[i] == label: #label != pred[i]
                fn +=1
                fn_indexes.append(i)
        stats = []
        stats.append(tp_indexes)
        stats.append(fp_indexes)
        stats.append(tn_indexes)
        stats.append(fn_indexes)
        domain_statistics[label] = stats
        
            
        try:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            #acc = (tp + tn) / (tp + tn + fp + fn)
        except ZeroDivisionError:
            value = float('Inf')
        if precision > 0 and precision != float('Inf') and recall > 0 and recall != float('Inf'):
            f1 = 2*((precision*recall)/(precision+recall))
            print(label.ljust(10),"P:", int(round(precision, 2)*100), "\t R:", int(round(recall, 2)*100), "".ljust(7), "F1:", int(round(f1, 2)*100))
        else:
            print(label.ljust(10),"P: Nan\t R: Nan \t F1: Nan")
       
        
    
        
        
    #MICROS AND MACROS
    #mircos
    micro_prec = precision_score(true, pred, average='micro')
    micro_rec = recall_score(true, pred, average='micro')
    micro_f1 = f1_score(true, pred, average='micro')
    #macros
    macro_prec = precision_score(true, pred, average='macro')
    macro_rec = recall_score(true, pred, average='macro')
    macro_f1 = f1_score(true, pred, average='macro')
    #average accuracy
    #acc = accuracy_score(true, pred)
    #printing
    print("".ljust(10),"--------------------------------------------------------------------------------")
    print("".ljust(10),"Micro P:", int(round(micro_prec, 2)*100), "\t Micro R:", int(round(micro_rec, 2)*100), "".ljust(5), "Micro F1:", int(round(micro_f1, 2)*100))
    print("".ljust(10),"Macro P:", int(round(macro_prec, 2)*100), "\t Macro R:", int(round(macro_rec, 2)*100), "".ljust(5), "Macro F1:", int(round(macro_f1, 2)*100))
    print("\n")


    return domain_statistics

    
    
# This and functions computs mutual information between one feature (= column of term_doc_matrix) and the labels
# The implementation of the function is based on:
# https://github.com/yftah89/structural-correspondence-learning-SCL/blob/master/pre.py
def GetTopNMI(X,labels):
    MI = []
    length = X.shape[1]
    
    for i in range(length):
        temp=mutual_info_score(X[:, i], labels)
        MI.append(temp)
    MIs = sorted(range(len(MI)), key=lambda i: MI[i])
    MIs.reverse()
    return MIs, MI


def get_emotion_words_NRC(labels_intersection):
    emotion_words = []
    
    with open("data/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt") as f:
        lines = f.read().split("\n")
        for i in range(len(lines)):
            line = lines[i].split("\t")
            if line[1] in labels_intersection and int(line[2]) == 1:
                #print(line[0], line[1], line[2])
                emotion_words.append(line[0])
                
    with open("data/NRC-Emotion-Lexicon-Senselevel-v0.92.txt") as f:
        lines = f.read().split("\n")
        for i in range(len(lines)):
            term = lines[i].split("\t")
            if term[1] in labels_intersection and int(term[2]) == 1:
                #print(line[0], line[1], line[2])
                terms = term[0].split("--")
                synonyms = terms[1].split(", ")
                emotion_words.append(terms[0])
                emotion_words.extend(synonyms)
    #print(len(set(emotion_words)), "emotion words were extracted from NRC as pivot and non-pivot features")
    return set(emotion_words)

    
# see the citations at the top of the code
if __name__ == '__main__':
    domain = []
    domain.append("isear") #0
    domain.append("tales-emotion") #1, 'disgust': 378
    domain.append("dailydialogues") #2, 'disgust': 353, 'fear': 174
    domain.append("tec") #3
    domain.append("crowdflower") #4, 'disgust': 179
    
    
    for src_domain_index, item_src in enumerate(domain, start=0):
        for trg_domain_index, item_trg in enumerate(domain, start=0):
            #GETTING DATA
            print("Getting data for source: ", domain[src_domain_index].upper(), "\n")
            src, src_labels = get_raw_data(domain[src_domain_index])
            train_src, train_src_labels, test_src, test_src_labels, unl_src = split_data_balanced(src, src_labels)

            print("\n")
            print("Getting data for target: ", domain[trg_domain_index].upper(), "\n")
            trg, trg_labels = get_raw_data(domain[trg_domain_index])
            train_trg, train_trg_labels, test_trg, test_trg_labels, unl_trg = split_data_balanced(trg, trg_labels)

            #CREATING VECTORIZER
            bigram_vectorizer = CountVectorizer(ngram_range=(1, 2), min_df=5, stop_words='english', binary=True)
            #TRAIN VECTORIZER
            X_2_train = bigram_vectorizer.fit_transform(train_src).toarray()
            #TEST VECTORIZER
            X_2_test = bigram_vectorizer.transform(test_trg).toarray()

            #TRAINING WITHOUT ADAPTATION
            logreg = LogisticRegression(C=0.1)
            logreg.fit(X_2_train, train_src_labels)

            print()
            print()
            print("################################################################################")
            print()
            if domain[src_domain_index] == domain[trg_domain_index]:
                print("IN-DOMAIN:") 
                y_pred_indomain = logreg.predict(X_2_test)
                print("TRAIN(" + domain[src_domain_index]+ " without DA)->TEST(in-domain): ")
                domain_stat_indomain = evaluate(test_trg_labels, y_pred_indomain)
                print()
                #print("TRAIN(" + domain[src_domain_index]+ " without DA)->TEST(in-domain): ")
                #print(domain_stat_indomain)
                print("#################################################################################")
                print()



            else:
                print("BASELINE:")
                logreg = LogisticRegression(C=0.1)
                logreg.fit(X_2_train, train_src_labels)
                y_pred_baseline = logreg.predict(X_2_test)
                print("TRAIN("+ domain[src_domain_index]+ " without DA)->TEST("+ domain[trg_domain_index] + "): ")
                domain_stat_baseline = evaluate(test_trg_labels, y_pred_baseline)
                print()
                #print("TRAIN("+ domain[src_domain_index]+ " without DA)->TEST("+ domain[trg_domain_index] + "): ")
                #print(domain_stat_baseline)
                print("################################################################################")
                print()



                print("CROSS-DOMAIN:")
                #GETTING UNLABELLED DATA
                unlabeled = train_src + test_src + unl_src + train_trg + test_trg + unl_trg
                #UNLABELED VECTORIZER
                bigram_vectorizer_unlabeled = CountVectorizer(ngram_range=(1, 2), min_df=5, stop_words='english', binary=True)
                X_2_train_unlabeled = bigram_vectorizer_unlabeled.fit_transform(unlabeled).toarray()
                #TEST VECTORIZER
                bigram_vectorizer_trg = CountVectorizer(ngram_range=(1, 2), min_df=5, stop_words='english', binary=True)
                X_trg = bigram_vectorizer_trg.fit_transform(train_trg).toarray()
                
                
                """
                #GETTING FREQUENT FEATURES
                cv = CountVectorizer(ngram_range=(1, 2), min_df=5, stop_words='english')
                cv_fit = cv.fit_transform(unlabeled)
                frequent_features = []
                freq_features = cv.get_feature_names()
                freqs = cv_fit.toarray().sum(axis=0)
                freq_dict = {k: v for k, v in zip(freq_features, freqs)}
                for key, value in sorted(freq_dict.items(), key=lambda item: (item[1], item[0]), reverse=True):
                    frequent_features.append(key)
                #EXTRACTING 500 PIVOTS
                pivotsCounts= []
                counter = 0
                for f in frequent_features:
                    if counter < 500:
                        if f in bigram_vectorizer.get_feature_names() and f in bigram_vectorizer_trg.get_feature_names():
                            pivotsCounts.append(bigram_vectorizer_unlabeled.get_feature_names().index(f))
                            counter+=1
                            
                
                
                #GETTING FREQUENT NRC FEATURES
                emo_dict = get_emotion_words_NRC(train_src_labels)
                cv = CountVectorizer(vocabulary=emo_dict)
                cv_fit = cv.fit_transform(unlabeled)
                frequent_features = []
                freq_features = cv.get_feature_names()
                freqs = cv_fit.toarray().sum(axis=0)
                freq_dict = {k: v for k, v in zip(freq_features, freqs)}
                for key, value in sorted(freq_dict.items(), key=lambda item: (item[1], item[0]), reverse=True):
                    frequent_features.append(key)
                #EXTRACTING 500 PIVOT FEATURES
                pivotsCounts= []
                counter = 0
                for f in frequent_features:
                    if counter < 500:
                        if f in bigram_vectorizer.get_feature_names() and f in bigram_vectorizer_trg.get_feature_names():
                            pivotsCounts.append(bigram_vectorizer_unlabeled.get_feature_names().index(f))
                            counter+=1
                """
                
                #GETTING FEATURES WITH HIGHEST MIs
                MIsorted, MIraw=GetTopNMI(X_2_train, train_src_labels)
                #MIsorted = MIsorted[0:1000]
                pivot_num = 500
                pivot_min_st = 10
                pivotsCounts= []
                names = []
                c=0
                i=0
                if len(MIsorted) < 500:
                    pivot_num = len(MI_sorted)
                while (c < pivot_num and i < len(MIsorted)):
                    name= bigram_vectorizer.get_feature_names()[MIsorted[i]]
                    if name in bigram_vectorizer_trg.get_feature_names():
                        names.append(name)
                        pivotsCounts.append(bigram_vectorizer_unlabeled.get_feature_names().index(name))
                        c+=1
                        #print("Feature is: ",name,". Its MI is: ",MIraw[MIsorted[i]])
                    i+=1
      
                         
                #EXTRACTING X,Y FOR PIVOT PREDICTION
                # y is the matrix only with pivots
                y = X_2_train_unlabeled[:,pivotsCounts]
                #print("Number of pivot features: ", y.shape[1])
                # x is the matrix with all non-pivot features excluded pivots
                x =np.delete(X_2_train_unlabeled, pivotsCounts, 1)
                #print("Number of non-pivot features: ", x.shape[1])

                #TRAIN PIVOT PREDICTORS
                start=datetime.now()
                print("\n")
                #print("Training pivot predictors ...\n")
                # extract all non-pivots
                inputs = x.shape[1]
                # create empty matrix to store trained weights in it. Row are pivots, columns are non-pivots. 
                pivot_mat = np.zeros((len(pivotsCounts), inputs))
                # for single pivot compute the correlation to each non-pivot
                for i in range(len(pivotsCounts)):
                    #clf = MultinomialNB(alpha=.01)
                    clf = linear_model.SGDClassifier(loss="modified_huber")
                    #clf = LogisticRegression(C=0.1)
                    if len([index for index, value in enumerate(y[:,i].tolist()) if value == 1]) == 0:
                        print(i, "th feature is empty \n")
                        break
                    clf.fit(x,y[:,i])
                    pivot_mat[i] = clf.coef_
                #print("finish traning")
                #necessary for dot product: Non-Pivots are in row in the tarin matrix and in columns in the pivot_mat(rix)
                pivot_mat = pivot_mat.transpose()
                svd50 = TruncatedSVD(n_components=50)
                pivot_mat50 = svd50.fit_transform(pivot_mat)
                #print("finished svd")
                mat = pivot_mat50
                print("Training time: ", datetime.now()-start)

                #TRAIN-REPR
                X_2_train_unl = bigram_vectorizer_unlabeled.transform(train_src).toarray()
                XforREP_src = np.delete(X_2_train_unl, pivotsCounts, 1)
                rep_src = XforREP_src.dot(mat)
                train_withDA = np.concatenate((X_2_train, rep_src), axis=1)
                #TARGET-REPR
                X_2_test_unl = bigram_vectorizer_unlabeled.transform(test_trg).toarray()
                XforREP_trg = np.delete(X_2_test_unl, pivotsCounts, 1)  
                rep_for_trg = XforREP_trg.dot(mat)
                test_withDA = np.concatenate((X_2_test, rep_for_trg), axis=1)
                #TRAINING WITH ADAPTATION
                logregDA =  LogisticRegression(C=0.1)
                logregDA.fit(train_withDA, train_src_labels)
                #APPLY AND EVALUATE
                y_pred_crossdomain = logregDA.predict(test_withDA)
                print("TRAIN("+ domain[src_domain_index]+ " with DA)->TEST("+ domain[trg_domain_index] + "): ")
                domain_stat_crossdomain = evaluate(test_trg_labels, y_pred_crossdomain)
                print()
                #print("TRAIN("+ domain[src_domain_index]+ " with DA)->TEST("+ domain[trg_domain_index] + "): ")
                #print(domain_stat_crossdomain)
                print("################################################################################")
                print()

    print("The End :-)")
