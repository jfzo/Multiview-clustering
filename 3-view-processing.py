from gensim.test.utils import common_texts
from gensim.utils import simple_preprocess
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import numpy as np
from gensim.models import LdaModel
from gensim.corpora import Dictionary

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
import nltk
import re
from gensim.utils import tokenize

import argparse
import sys
import scipy
#### Utility functions to customize document tokenization

#stemmer = PorterStemmer()

def pre_process(text):
    
    # lowercase
    text=text.lower()
    
    #remove tags
    text=re.sub("<!--?.*?-->","",text)
    
    # remove special characters and digits
    text=re.sub("(\\d|\\W)+"," ",text)
    
    return text

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    #stems = stem_tokens(tokens, stemmer)
    stems = tokens
    return stems

def gensim_tokenizer(text):
    return list(tokenize(text, deacc=True, lowercase = True))

#_fileName = ".."
# _VSize : Size of document vectors (skipgram)
#_Ntopics : Number of topics to use in LDA
#_mat_outputfile : file name that stores the output data-representation matrix 


""" How to call this script:
>> python3 3-view-processing.py --i ../data/reuters-r8/r8-test.csv --o ../data/reuters-r8/r8-test-mat-skipgram.npz --type skipgram
>> python3 3-view-processing.py --i ../data/reuters-r8/r8-test.csv --o ../data/reuters-r8/r8-test-mat-tfidf.npz --type tfidf
>> python3 3-view-processing.py --i ../data/reuters-r8/r8-test.csv --numtopics 8 --o ../data/reuters-r8/r8-test-mat.npz --type lda
"""

parser = argparse.ArgumentParser(description='Builds a specific representation for text data!')
parser.add_argument("--i", type=str, help="Location of input file with one doc per line.")
parser.add_argument("--vecsize", default=100, type=int, help="Size of output vectors in the Skipgram representation.")
parser.add_argument("--numtopics", default=10, type=int, help="Number of topics to consider for the LDA representation.")
parser.add_argument("--o", type=str, help="Location of output file containing the document-feature matrix.")
parser.add_argument("--type", choices=["skipgram", "tfidf", "lda"], required=True, type=str, help="Type of document representation to build.")
parser.add_argument('--stopwords', default="None", type=str, help='Use stopwords for TF-IDf. Not using by default.')

args = parser.parse_args()
                             
_fileName = args.i
_VSize = args.vecsize
_Ntopics = args.numtopics
_mat_outputfile = args.o
        
_use_skipgram = False
_use_LDA = False
_use_TFIDF = False

_use_stopwords = None

if args.stopwords != "None":
    _use_stopwords = args.stopwords

if args.type ==  "skipgram":
    _use_skipgram = True
elif args.type ==  "lda":
    _use_LDA = True
elif args.type ==  "tfidf":
    _use_TFIDF = True
else:
    print("You must pick one of the type options.")
    sys.exit(-1)
        

tokenized_dataset = list()

if _use_skipgram or _use_LDA:
    with open(_fileName) as f:
        for line in f:
            tokenized_dataset.append(simple_preprocess(line)) # Convert a document into a list of lowercase tokens, ignoring tokens that are too short or too long
else:
    with open(_fileName) as f:
        for line in f:
            tokenized_dataset.append(pre_process(line)) # adding doc
            #tokenized_dataset.append(simple_preprocess(line)) # adding doc
    

# Skipgram representation
if _use_skipgram:
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(tokenized_dataset)]
    model = Doc2Vec(vector_size=_VSize, min_count=2, epochs=40)

    #Build a vocabulary
    model.build_vocab(documents)

    #train the model
    model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)

    model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)

    # get the vector representation for each document
    N = len(tokenized_dataset)
    sD = np.zeros((N, model.vector_size))

    for i,d in enumerate(tokenized_dataset):
        sD[i,:] = model.infer_vector(d)
        
    np.savez(_mat_outputfile, sD)

elif _use_LDA:

    # Create a dictionary representation of the documents.
    dictionary = Dictionary(tokenized_dataset)
    dictionary.filter_extremes(no_below=10, no_above=0.8)

    corpus = [dictionary.doc2bow(doc) for doc in tokenized_dataset]

    print('Number of unique tokens: %d' % len(dictionary))
    print('Number of documents: %d' % len(corpus))
    # Train LDA model.
    # Set training parameters.
    num_topics = _Ntopics
    chunksize = 2000
    passes = 20
    iterations = 400
    eval_every = None  # Don't evaluate model perplexity, takes too much time.

    # Make a index to word dictionary.
    temp = dictionary[0]  # This is only to "load" the dictionary. f not executed dictionary seems empty!
    id2word = dictionary.id2token

    model = LdaModel(
        corpus=corpus,
        id2word=id2word,
        chunksize=chunksize,
        alpha='auto',
        eta='auto',
        iterations=iterations,
        num_topics=num_topics,
        passes=passes,
        eval_every=eval_every
    )
    
    N = len(tokenized_dataset)
    ldaD = np.zeros((N, model.num_topics))

    for i,d in enumerate(tokenized_dataset):
        doc_i = dictionary.doc2bow(tokenized_dataset[i])
        ldaD[i,:] = np.array([prob_topic for _, prob_topic in model.get_document_topics(doc_i, minimum_probability=0)])
    
    np.savez(_mat_outputfile, ldaD)

else: # Tf-IDF

    stemmer = PorterStemmer()

    vectorizer = TfidfVectorizer(max_df=0.8, max_features=40000, min_df=0.05, stop_words=_use_stopwords, use_idf=True, tokenizer=tokenize)
    tfidfdata = vectorizer.fit_transform(tokenized_dataset)
    
    scipy.sparse.save_npz(_mat_outputfile, tfidfdata)
    
    #np.savez(_mat_outputfile, tfidfdata)