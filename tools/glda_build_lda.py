import json
import re
import requests
import time
import warnings
import gensim
import guidedlda
import logging
from gensim import interfaces, utils, matutils

from funcy.colls import none

'''
Created on Mar 9, 2018

@author: wilson.penha
'''
import pyLDAvis.sklearn
import os
import base64
import codecs
import numpy as np
import pandas as pd

from docutils.nodes import inline

from collections import Counter
from scipy import int64
from scipy.misc import imread
from scipy.sparse import (csr_matrix, lil_matrix, coo_matrix)

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.externals import joblib
from sklearn.externals.joblib import Parallel, delayed
from sklearn.utils import (check_random_state, check_array,
                     gen_batches, gen_even_slices, _get_n_jobs)
from nltk.corpus import stopwords
 
import json 
import nltk
from cgitb import text

import itertools

from tools.my_display import MyDisplay
from tools.text import LabelCountVectorizer
from tools.label_finder import BigramLabelFinder
from tools.label_ranker import LabelRanker
from tools.pmi import PMICalculator
from tools.corpus_processor import (
                                        CorpusWordLengthFilter,
                                        CorpusPOSTagger,
                                        CorpusStemmer)
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

from nltk.stem import WordNetLemmatizer

CURDIR = os.path.dirname(os.path.realpath(__file__))

logger = logging.getLogger('guidedlda')
logger.propagate = False

class LemmaCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        lemm = WordNetLemmatizer()
        analyzer = super(LemmaCountVectorizer, self).build_analyzer()
        return lambda doc: (lemm.lemmatize(w) for w in analyzer(doc))

class TopicModelingTrainer(object):
    def __init__(self):
        self.tf_feature_names=[]

    # Define helper function to print top words
    def build_email_corpus(self, body):
        isBody=True
        newBody=''
        for line in body.splitlines():
            if line.find('----Original Message----')>=0:
                isBody=False
            if line.find('Message-ID:')>=0:
                isBody=False
            if line.find('Mime-Version: ')>=0:
                isBody=False
            if line.find('To: ')>=0:
                isBody=False
            if line.find('From: ')>=0:
                isBody=False
            if line.find('cc: ')>=0:
                isBody=False
                
            if line.find('----Original Message----')>=0:
                isBody=False
                
            if line.find('Subject:')>=0:
    #             print(line)
                isBody=True
            
            if isBody:    
                line = re.sub(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+.[a-zA-Z0-9-]+"," ",line.lower())
                line = re.sub(r"\b(https?://)?(www\.)?([a-z0-9\-.]+)(?:\.[a-z\.]+[\/]?)\w+/(?:\w+/)+(\S+)\b"," ", line)
                line = re.sub(r"(^|\W)\d+", " ", line)
                replaces = {"Subject:":"","\"":" ",",":" ",".":" ","-":" ",":":" ","/":" ","$":"","(":"",")":"","*":"","!":"",
                            "[":"","]":"","=":"","&":"","'":"","`":"","#":"","_":"","@":"","error occurred attempting initialize borland database engine error":"",
                            ";":"",">":"","<":"","?":"","  ":" ","\\":" ","\n":" ","\t":" ","%":" ","\'":" ","\"":" ","—":" "}
                line = self.replace_all(line, replaces)
                line = re.sub(r'\W*\b\w{1,2}\b', " ", line)
                newBody += line+' '
                
            if line.find("X-FileName: ")>=0:
                isBody=True
    
        return newBody
        
    def replace_all(self, text, dic):
        for i, j in dic.items():
            text = text.replace(i, j)
        return text
    
    def find_labels(self, n_labels, label_min_df, tag_constraints, tagged_docs, n_cand_labels, docs):
        cand_labels = []
        while (len(cand_labels)<n_labels):
    #             print("Generate candidate bigram labels(with POS filtering)...")
            finder = BigramLabelFinder('pmi', min_freq=label_min_df,
                                       pos=tag_constraints)
    
            if tag_constraints:
                cand_labels = finder.find(tagged_docs, top_n=n_cand_labels)
            else:  # if no constraint, then use untagged docs
                cand_labels = finder.find(docs, top_n=n_cand_labels)
        
            print("Finding Collected {} candidate labels".format(len(cand_labels)))
            if len(cand_labels)>=n_labels:
                break
            else:
                label_min_df -= 1
                if (label_min_df<1):
                    # build tags based on Singular Noun, Noun and Adjetive, Noun
                    label_tags = ['NNS,NN', 'NN,NN', 'JJ,NN']
                    label_min_df = 5
                    tag_constraints = []
                    for tags in label_tags:
                        tag_constraints.append(tuple(map(lambda t: t.strip(),
                                                             tags.split(','))))
    
        
        return cand_labels
        
    def load_stopwords(self):
        with codecs.open(CURDIR + '/../resources/stopwords_en.txt', mode='r',encoding='utf8') as f:
            return map(lambda s: s.strip(),
                       f.readlines())
    # Define helper function to print top words
    def build_top_words(self, model, feature_names, n_top_words):
        for index, topic in enumerate(model.components_):
            topic_name = "Topic_#{}:".format(index)
    #             message += " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1 :-1]])
            topic_json = {"topic" : topic_name,
                          "terms" : " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1 :-1]])  
                }
            self.topics["topics"].append(topic_json)
    #             print(topic_json)
    #         print("="*70)
    
    def print_top_words(self, model, feature_names, n_top_words):
        for i, topic in enumerate(model.components_):
            topic_ = topic
            topic_ = topic_ / topic_.sum()  # normalize to probability distribution
            bestn = matutils.argsort(topic_, n_top_words, reverse=True)
            topic_ = [(feature_names[id], topic_[id]) for id in bestn]
            topic_ = ' + '.join(['%.3f*"%s"' % (v, k) for k, v in topic_])
            print("Topic#",i,":",topic_)

#             message = "Topic #%d: " % topic_idx
#             message += " ".join([feature_names[i]
#                                  for i in topic.argsort()[:-n_top_words - 1:-1]])
#             print(message)
#         print()
    
    def save_lda_topic_model(self):
        joblib.dump(self.lda, 'glda_topic_model_small.lda') 
        
    def load_lda_topic_model(self):
        self.lda = joblib.load('glda_topic_model_small.lda')
    
    def save_term_vector_topic_model(self, vectorizer):
        joblib.dump(vectorizer, 'glda_vectorizer_topic_model_small.vtf') 
        
    def load_term_vector_topic_model(self):
        vectorizer = joblib.load('glda_vectorizer_topic_model.vtf')
        return vectorizer
    
    def load_fitted_term_vector(self):
        ftf = joblib.load('glda_vectorizer_topic_model_small.ftf')
        return ftf
    
    def label_relevance(self, label_tokens, context_tokens, topic_id):
        """
        Calculate the relevance position that the label appears
        in the context of the topics words frequency
        
        Parameter:
        ---------------

        label_tokens: list|tuple of str
            the label tokens
        context_tokens: list|tuple of str
            the sentence tokens

        Return:
        -----------
        int: the label frequency in the sentence
        """
        label_len = len(label_tokens)
        cnt = 0
        
        pos = []
        for i in range(0,len(context_tokens) - label_len + 1):
            for j in range(0,label_len):
                if label_tokens[j].lower() == context_tokens[i+j].lower():
                    pos.append(i+1)
        
        relevance_index=int(sum(pos)/2)
        
#         print({"topic_id":topic_id, "relevance_score": relevance_index, "topic_label":label_tokens})
        return relevance_index

    def process_corpus(self):
        headers = {
            'Content-Type': 'application/json',
        }
        
        params = {}
        
        serialIds = []
        
        for i in range(1,2000):
            serialIds.append(i+1)
            
        query = {"archiveId" : "5c7d04b9-a4a0-4518-97d6-e011ba8a28d1", 
            "serialIds":serialIds
        }
        
    #     print('query ', query)
    
        start = time.time()
    
        url = 'http://localhost:8080/msg'
    #     url = 'http://dldev6-test.office.globalrelay.net:9200/messages/_search'
        response = requests.post(url, headers=headers, data=json.dumps(query))
        response.raise_for_status()
    
        emailthread_json = json.loads(response.content)
        
        end = time.time()
        elapsed = end - start
        print(elapsed)
    
        start = time.time()
        
        corpus = []
        for emails in emailthread_json['response']:
            if not emails['subject'] == None:
                subject = 'Subject: '+emails['subject']
            else:
                subject = 'Subject: '
                
            body = subject + '\n' +emails['body']
            corpus.append(self.build_email_corpus(body.replace("Group", "")))
        
        docs = []
        for l in corpus:
            sents = nltk.sent_tokenize(l.strip())
            docs.append(list(itertools.chain(*map(
                nltk.word_tokenize, sents))))
        return docs
        
    def build_lda(self, n_topics=10):

        n_components = n_topics
        n_top_words = 30     

        
        tf_vectorizer = self.load_term_vector_topic_model()
        
        print("Building the Vectorizer for Topic model...")
        tf = tf_vectorizer

        self.tf_feature_names = tf_vectorizer.get_feature_names()
        
        # test semi-supervided Topics
        print("Building the Alternative LDA Topic model...")
        vocab = tuple(self.tf_feature_names)
        
        seed_topic_list = {}
        word2id = {}
        
        if seed_topic_list.items().__sizeof__()>0: 
            word2id = dict((v, idx) for idx, v in enumerate(tuple(self.tf_feature_names)))
        
        array_tf = tf.toarray()
         
        # Guided LDA with seed topics.
#         seed_topic_list = {'percent': 7, 'year': 7, 'month': 7, 'quarter': 7} 
        seed_topics = {}
        for term, topic in seed_topic_list.items():
            seed_topics[word2id[term]] = topic
            
        model = guidedlda.GuidedLDA(n_topics=n_components, n_iter=100, random_state=7, refresh=50)
        
        logger.propagate = False
        
        model.fit(array_tf, seed_topics=seed_topics, seed_confidence=0.15)
        
#         model.fit(array_tf)
        
        self.lda = model
         
        print("Printing the Alternative LDA Topic model...")
        topic_word = model.topic_word_
        n_top_words = 30
        
        # print topics with words and score rank
        for i, topic_dist in enumerate(topic_word):
#             topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
#             print('Topic {}: {}'.format(i, ' '.join(topic_words)))
            topic_ = topic_dist
            topic_ = topic_ / topic_.sum()  # normalize to probability distribution
            bestn = matutils.argsort(topic_, 30, reverse=True)
            topic_ = [(self.tf_feature_names[id], topic_[id]) for id in bestn]
            topic_ = ' + '.join(['%.3f*"%s"' % (v, k) for k, v in topic_])
            print("Topic#",i,":",topic_)

        
        py_glda_vis = MyDisplay()
        
        list_topic_names = ['T00','T01','T02','T03','T04','T05','T06','T07','T08','T09',
                            'T10','T11','T12','T13','T14','T15','T16','T17','T18','T19',
                            'T20','T21','T22','T23','T24','T25','T26','T27','T28','T29',
                            'T30','T31','T32','T33','T34','T35','T36','T37','T38','T39']
        
        list_topic_labels = []
        
        visualization = py_glda_vis.prepare_glda(model, tf, tf_vectorizer, mds='tsne')
        
        # processing the labels in the same order of topic relevance
        print(list(visualization[6:])[0])
        
        for i,topic in enumerate(list(visualization[6:])[0]):
            list_topic_labels.append(list_topic_names[topic-1])
            
        topic_name = {"topic.names" : list_topic_labels}
        
        print(topic_name)
        
#         visualization_html = py_lda_vis.prepared_data_to_html(visualization,
#                                                               json_names=topic_name)
        py_glda_vis.save_html(visualization, 
                             'GLDA_Visualization_labels.html',
                             json_names=topic_name)
        
        py_glda_vis.save_html(visualization, 
                             'GLDA_Visualization_nolabels.html')
# this example is to be used for future Topic Model builder 
#         try:
#             self.load_lda_topic_model() 
#         except FileNotFoundError:
            # if no lda topic_model file, build a new one
        self.save_lda_topic_model() 
        
#         self.encoded_html = base64.b64encode(visualization_html.encode())
        self.encoded_html = ''
        
        self.topics = {}

    #         with open('corpus_data.txt', 'w') as outfile:
    #             json.dump(json.dumps(corpus,indent=1), outfile)    
    def load_line_corpus(self, path, tokenize=True):
        docs = []
        
        print('Building the corpus...')
        
        stop = list(self.load_stopwords())
        stop.append('off')
        stop.append('http')
        stop.append('www')
        stop.append('edt')
        stop.append('est')
        stop.append('mdt')
        stop.append('pst')
        stop.append('pt')
        
        with codecs.open(path, mode="r", encoding="utf8") as f:
            for l in f:
                if tokenize:
                    line = re.sub(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+.[a-zA-Z0-9-]+"," ",l.lower())
                    line = re.sub(r'[^\x00-\x7f]',r' ', line) 
                    line = re.sub(r"\b(https?://)?(www\.)?([a-z0-9\-.]+)(?:\.[a-z\.]+[\/]?)\w+/(?:\w+/)+(\S+)\b"," ", line)
                    line = re.sub(r"(^|\W)\d+", " ", line)
                    replaces = {"Subject:":"","\"":" ",",":" ",".":" ","-":" ",":":" ","/":" ","$":"","(":"",")":"","*":"","!":"",
                                "[":"","]":"","=":"","&":"","'":"","`":"","#":"","_":"","@":"","error occurred attempting initialize borland database engine error":"",
                                ";":"",">":"","<":"","?":"","  ":" ","\\":" ","\n":" ","\t":" ","%":" ","\'":" ","\"":" ","—":" "}
                    line = self.replace_all(line, replaces)
                    line = re.sub(r'\W*\b\w{1,2}\b', " ", line)
                    
                    sents = nltk.sent_tokenize(line.strip())
                    sents = list(itertools.chain(*map(nltk.word_tokenize, sents)))
                    sents = [i for i in sents if i not in stop]
#                     lanste = LancasterStemmer()
#                     sents = [lanste.stem(i) for i in sents]
        
                    docs.append(sents)
                else:
                    docs.append(l.strip())

        return docs

if __name__ == "__main__":
    tmt = TopicModelingTrainer()
#     tmt.build_lda(tmt.process_corpus(),20)
    tmt.build_lda(35)
        
    
        
        


     