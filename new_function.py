import numpy
import itertools
from nltk import word_tokenize, sent_tokenize, pos_tag
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from collections import Counter
from isf_func import *
from cosine import *
import scipy.linalg as SL
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import networkx as nx
from find_para import *
def new_function(U,k_input,type_of_graph,sentence_form,in_sent,dimen,similarity_threshold):
    k_value = []
    for i in range(k_input):
        k_value.append(i)
    extract = U[:,k_value]   #Feature matrix obtained after applying SVD on previous Feature Matrix and Selecting k columns.
    similarity = cosine_similarity(extract)  #Similarity matrix containing similarity b/w each pair nodes.
    if type_of_graph == 'FD':                  #Constructing similarity matrix in case Forward directed graph
        for i in range(dimen):
            for j in range(dimen):
                if i > j :
                    similarity[i,j] = 0
                if similarity[i,j] < similarity_threshold:
                    similarity[i,j] = 0

    if type_of_graph == 'BD':                  #Constructing similarity matrix in case Backward directed graph
        for i in range(dimen):
            for j in range(dimen):
                if i < j :
                    similarity[i,j] = 0
                if similarity[i,j] < similarity_threshold:
                    similarity[i,j] = 0

    if type_of_graph == 'UD':                  #Constructing similarity matrix in case Undirected graph
        for i in range(dimen):
            for j in range(dimen):
                if similarity[i,j] < similarity_threshold:
                    similarity[i,j] = 0
    
    G=nx.from_numpy_matrix(similarity,create_using=nx.DiGraph())   #Graph object created from similarity matrix.
    pr = nx.pagerank(G, alpha=0.15) #Finds the PageRank of Graph
    st = Counter(pr)
    op = []
    for k,v in st.most_common(int(dimen/5)):  #Compression Rate 20%,i.e, (no. of sentences/5)
        op.append(k)
    sent_in_para = []
    for i in range(len(op)):
        #k = find_para(sentence_form[op[i]],in_sent)
        k = find_para(sentence_form[op[i]],in_sent)
        sent_in_para.append(k)
    f2 = "Summary for k = " + str(k_input) +".txt"
    with open(f2, 'a') as f:
        '''
        print('\nSummary when k and similarity threshold is taken as ->',k_input,' and ',similarity_threshold,file=f)
        if type_of_graph == 'UD':
            print('And type of graph is Undirected.',file=f)
        if type_of_graph == 'FD':
            print('And type of graph is Forward directed.',file=f)
        if type_of_graph == 'BD':
            print('And type of graph is Backward directed.',file=f)
        print('\nSummary is-->',file=f)
        '''
        for i in range(len(op)):
            #print(op[i]+1,".",sentence_form[op[i]],file=f)
            print(sentence_form[op[i]],file=f)
    return sent_in_para

