import gensim
import numpy as np
import nltk
import datetime

from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel, ldamodel
from gensim.models.wrappers import LdaMallet
from gensim.corpora import Dictionary
from stop_words import get_stop_words
from nltk.stem.lancaster import LancasterStemmer
from tools.my_display import MyDisplay

import pyLDAvis.gensim
import pyLDAvis.sklearn
from sklearn.externals import joblib
import os, re, operator, warnings
import codecs
import itertools
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

from nltk.stem import WordNetLemmatizer

class TopicModelingTrainer(object):
    def __init__(self):
        self.tf_feature_names=[]

    def replace_all(self, text, dic):
        for i, j in dic.items():
            text = text.replace(i, j)
        return text

    def save_lda_topic_model(self, ldamodel):
        joblib.dump(ldamodel, '/data/gensim_topic_model.lda') 
        
    def load_lda_topic_model(self):
        ldamodel = joblib.load('/data/gensim_topic_model.lda')
        return ldamodel
    
    def save_term_vector_topic_model(self, dictionary):
        joblib.dump(dictionary, '/data/dictionary_topic_model.dic') 
        
    def load_term_vector_topic_model(self):
        dictionary = joblib.load('/data/dictionary_topic_model.dic')
        return dictionary
    
    def load_line_corpus(self, path, tokenize=True):
        docs = []
        
        print('Building the corpus...')
        
        now = datetime.datetime.now()
        time_now = (now.year, now.month, now.day, now.hour, now.minute, now.second)
        print(time_now)

        stop = stopwords.words('english')
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
        now = datetime.datetime.now()
        time_now = (now.year, now.month, now.day, now.hour, now.minute, now.second)
        print(time_now)

        return docs

    def build_lda(self, docs):
        docs = docs
        
        bigram = gensim.models.Phrases(docs)
        
        print('Building BiGrams from the corpus...')
        now = datetime.datetime.now()
        time_now = (now.year, now.month, now.day, now.hour, now.minute, now.second)
        print(time_now)
        texts = [bigram[line] for line in docs]
        now = datetime.datetime.now()
        time_now = (now.year, now.month, now.day, now.hour, now.minute, now.second)
        print(time_now)
        
        dictionary = Dictionary(texts)
        print('Building the dictionary for the corpus...')
        now = datetime.datetime.now()
        time_now = (now.year, now.month, now.day, now.hour, now.minute, now.second)
        print(time_now)
        corpus = [dictionary.doc2bow(text) for text in texts]
        now = datetime.datetime.now()
        time_now = (now.year, now.month, now.day, now.hour, now.minute, now.second)
        print(time_now)
        
        self.save_term_vector_topic_model(dictionary)
        
#         print('Building LSI model ...')
#         now = datetime.datetime.now()
#         time_now = (now.year, now.month, now.day, now.hour, now.minute, now.second)
#         print(time_now)
#         lsimodel = LsiModel(corpus=corpus, num_topics=20, id2word=dictionary)
#         lsimodel.show_topics(num_topics=20)
#         now = datetime.datetime.now()
#         time_now = (now.year, now.month, now.day, now.hour, now.minute, now.second)
#         print(time_now)
        
#         print('Building HDP model ...')
#         now = datetime.datetime.now()
#         time_now = (now.year, now.month, now.day, now.hour, now.minute, now.second)
#         print(time_now)
#         hdpmodel = HdpModel(corpus=corpus, id2word=dictionary)
#         hdpmodel.show_topics()
#         now = datetime.datetime.now()
#         time_now = (now.year, now.month, now.day, now.hour, now.minute, now.second)
#         print(time_now)
#         
        print('Building LDA model ...')
        now = datetime.datetime.now()
        time_now = (now.year, now.month, now.day, now.hour, now.minute, now.second)
        print(time_now)
        ldamodel = LdaModel(corpus=corpus, num_topics=15, id2word=dictionary, iterations=100, passes=50)
        print(ldamodel.show_topics(num_topics=15,num_words=30))
        now = datetime.datetime.now()
        time_now = (now.year, now.month, now.day, now.hour, now.minute, now.second)
        print(time_now)
        
        self.save_lda_topic_model(ldamodel)
        
        print('Building LDA model Visualization ...')
        now = datetime.datetime.now()
        time_now = (now.year, now.month, now.day, now.hour, now.minute, now.second)
        print(time_now)
        visualization = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)

        py_lda_vis = MyDisplay()
        
        list_topic_names = ['T00','T01','T02','T03','T04','T05','T06','T07','T08','T09',
                            'T10','T11','T12','T13','T14','T15','T16','T17','T18','T19']
        
        list_topic_labels = []
        
        # processing the labels in the same order of topic relevance
        print(list(visualization[6:])[0])
        
        for i,topic in enumerate(list(visualization[6:])[0]):
            list_topic_labels.append(list_topic_names[topic-1])
            
        topic_name = {"topic.names" : list_topic_labels}
        
        print(topic_name)
        
        py_lda_vis.save_html(visualization, 
                             '/data/LDA_Gensim_Financial_labels.html',
                             json_names=topic_name)
        
        py_lda_vis.save_html(visualization, 
                             '/data/LDA_Gensim_Financial_nolabels.html')

#         pyLDAvis.save_html(visualization,'/data/LDA_gesim.html')
        now = datetime.datetime.now()
        time_now = (now.year, now.month, now.day, now.hour, now.minute, now.second)
        print(time_now)

#         lsitopics = [[word for word, prob in topic] for topicid, topic in lsimodel.show_topics(formatted=False)]
        
#         hdptopics = [[word for word, prob in topic] for topicid, topic in hdpmodel.show_topics(formatted=False)]
        
        ldatopics = [[word for word, prob in topic] for topicid, topic in ldamodel.show_topics(formatted=False)]
        
#         lsi_coherence = CoherenceModel(topics=lsitopics[:20], texts=texts, dictionary=dictionary, window_size=10).get_coherence()
#         hdp_coherence = CoherenceModel(topics=hdptopics[:20], texts=texts, dictionary=dictionary, window_size=10).get_coherence()
        print('Building LDA model Coherence ...')
        now = datetime.datetime.now()
        time_now = (now.year, now.month, now.day, now.hour, now.minute, now.second)
        print(time_now)
        lda_coherence = CoherenceModel(topics=ldatopics, texts=texts, dictionary=dictionary, window_size=10).get_coherence()
        now = datetime.datetime.now()
        time_now = (now.year, now.month, now.day, now.hour, now.minute, now.second)
        print(time_now)


#         print("LSI Coherence : "+repr(lsi_coherence))
#         print("HDP Coherence : "+repr(hdp_coherence))
        print("LDA Coherence : "+repr(lda_coherence))

        print("I'm almost there!")

if __name__ == "__main__":
    tmt = TopicModelingTrainer()
#     tmt.build_lda(tmt.process_corpus(),20)
    tmt.build_lda(tmt.load_line_corpus("/data/corpus/1k_reuters.txt"))
