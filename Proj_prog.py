"""
This is implementation for finding Summary of a Document. 
@authors Rahul Dogra
"""
import nltk
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
import matplotlib.pyplot as plt
from find_para import *

#**************************** Functions for Feature Matrix -- Start **************************************************
#Function to find segment IDs normalized by total no. of segments {for Feature 1}
def segid(length):
   ID = []
   for i in range(length):
      ids = (i+1)/length
      ID.append(ids)
   return ID

#Function to find paragraph IDs normalized by the total no. of paragraphs {for Feature 2}
def paraid(in_sent, length):
   ID = []
   for i in range(length):
      for j in range(len(in_sent[i])):
         ids = (i+1)/length
         ID.append(ids)
   return ID

#Function to find segment offset within the paragraph containing it {for Feature 3}
def paraoff(in_sent, length):
   P_Off = []
   for i in range(length):
      for j in range(len(in_sent[i])):
         offsets = (j+1)/len(in_sent[i])
         P_Off.append(offsets)
   return P_Off

#Function to find location of the paragraph containing the segment relative to the document {for Feature 4}
def paraloc(para_id, length):
   Locn = []
   for i in range(length):
      if para_id[i]<0.2 :
         l = 1
      elif para_id[i]<0.8 :
         l = 2
      else :
         l = 3
      Locn.insert(i,l)
   return Locn

#Function to find segment location in the document {for Feature 5}
def segloc(seg_id):
   Locn = []
   for i in range(len(seg_id)):
      if seg_id[i]<0.25 :
         l = 1
      elif seg_id[i]<0.50 :
         l = 2
      elif seg_id[i]<0.75 :
         l = 3
      else :
         l = 4
      Locn.insert(i,l)
   return Locn

#Function to find segment length normalized by the length of the longest segment {for Feature 6}
def seglen(max_seg_len, stemmed_list):
   seg_len = [];
   for i in range(len(stemmed_list)):
      x = (len(stemmed_list[i]))/(max_seg_len)
      seg_len.append(x)
   return seg_len

#Function to find the no. of title words that appear in the segment normalized by total no. of title words {for Feature 7}
def titleword(stemmed_list, length):
   titlewords = []
   for i in range(len(stemmed_list)):
      count=0
      for j in range(length):
         count += stemmed_list[i].count(titles[j])
      x = count/length
      titlewords.append(x)
   return titlewords
#**************************** Functions for Feature Matrix --End-- **************************************************  


print("\nGetting Document Summary")
input("Press any key to resume")

file_name = input('\nEnter the text file name with extension -->>')  #Inputting file name

f = open(file_name)   #Open File 
text = f.read()       #Reading File

title_name = input('\nEnter the title of document -->>') #Inputting Title

k_input = int(input('\nEnter the value of K -->>'))   #Inputting parameter k

similarity_threshold = float(input('\nEnter the similarity threshold -->')) #Inputting Similarity Threshold

type_of_graph = input('\nEnter the type of graph :: \nUndirected(UD) or Forward Directed(FD) or Backward Directed(BD) -->> ')


#************************   Pre Processing Start  **************************
in_para = text.split('.\n')  #Split the document into paragraphs

in_sent = []
for i in range(len(in_para)):
        sent = sent_tokenize(in_para[i]) #Sentence tokenizer
        in_sent.append(sent)
#print("\nSentences are:",in_sent)
sentence_form = sent_tokenize(text)
dimen = len(sentence_form)

   

in_words = []
for i in range(len(in_sent)):
        for j in range(len(in_sent[i])):
                words = nltk.word_tokenize(in_sent[i][j])  #Word tokenizer for sentences
                in_words.append(words)
title_tokenize = nltk.word_tokenize(title_name) #Word tokenizer for title.

#print("\nWords are : ",in_words)
#******* Stop Word Removal *******
stop_words = set(stopwords.words('english'))
stop_words.update(['.',',','"',"'",'?','!',':',';','(',')','[',']','{','}'])

title = []
for i in range(len(title_tokenize)):
        titlm = title_tokenize[i].lower()
        if titlm not in stop_words:
           filtrs = titlm
           title.append(filtrs)


filter_words = []
for i in range(len(in_words)):
        filtr = [word.lower() for word in in_words[i] if word.lower() not in stop_words]
        filter_words.append(filtr)

#***** Stemming Start ******

porter_stemmer = PorterStemmer()
stemmed_words = []
for i in range(len(filter_words)):
        for j in range(len(filter_words[i])):
                stemm = porter_stemmer.stem(filter_words[i][j])
                stemmed_words.append(stemm)
titles = []
for j in range(len(title)):
         stemm = porter_stemmer.stem(title[j])
         titles.append(stemm)

#***** Stemming End ********

stemmed_list = []
iv= 0
for i in range(len(filter_words)):
        st = len(filter_words[i])
        stemmed_list.append(stemmed_words[iv:st+iv])
        iv+=st
#print("\nStemmed document list --> ", stemmed_list)


#*************************   Pre Processing End  *****************************

#*********************  Feature Generation Start *****************************
#Feature 1
seg_id = segid(len(stemmed_list))
#print("\nSegment ids --> ", seg_id)

#Feature 2
para_id = paraid(in_sent, len(in_para))
#print("\nParagraph ids for segments --> ", para_id)

#Feature 3
para_off = paraoff(in_sent, len(in_para))
#print("\nParagraph offset for segments --> ", para_off)

#Feature 4
para_loc = paraloc(para_id, len(para_id))
#print("\nParagraph location for segments --> ", para_loc)

#Feature 5
seg_loc = segloc(seg_id)
#print("\nSegment location for segments --> ", seg_loc)

#Feature 6
max_seg_len = len(max(stemmed_list, key = len))
seg_len = seglen(max_seg_len, stemmed_list)
#print("\nSegment length --> ", seg_len)

#Feature 7
title_word = titleword(stemmed_list, len(titles))
#print("\nNumber of title words that appear in the segment --> ", title_word)

#Feature 8
b_val = Counter(titles)
#print("\nStemmed List->>>>",stemmed_list)

cos_seg_title = cosine(stemmed_list,b_val)
#print("\nCosine similarities between title and segments --> ", cos_seg_title)

s_val = Counter(stemmed_words)
cos_seg_doc = cosine(stemmed_list,s_val);

#Feature 9
total_tf = []
for i in range(len(stemmed_list)):
   val = Counter(stemmed_list[i])
   words  = list(val.keys())                      #keys() returns keys
   vect = [val.get(word, 0) for word in words]    #get() returns a value for the given key
   sum = 0
   for j in range(len(vect)):
      sum += vect[j]
   total_tf.append(sum)
#print("\nTotal term frequency for segements --> ", total_tf)

#Feature 10
average_tf = []
for i in range(len(stemmed_list)):
	val = Counter(stemmed_list[i])
	leng  = len(list(val.keys()))
	average_tf.append((total_tf[i])/leng)
#print("\nAverage term frequency for segements --> ", average_tf)

#Feature 11
l = [item for sublist in in_sent for item in sublist]
bloblist = []
for i in range(len(in_sent)):
   for j in range(len(in_sent[i])):
      fg = tb(in_sent[i][j])
      bloblist.append(fg)
a = []
for i, blob in enumerate(bloblist):
	scores = {word: tfidf(word, blob, bloblist) for word in blob.words}
	sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
	s = sorted_words
	a.append(s)

sum_isf = []
for i in range(len(bloblist)):
	sum = 0;
	for j in range(len(a[i])):
		sum += a[i][j][1]
	sum_isf.append(sum)
#print("\nTotal tf * isf -->",sum_isf)

#Feature 12
mat = []
for i in range(len(in_words)):
	st = sum_isf[i] / len(in_words[i])
	mat.append(st)
#print("\nAverage tf * isf --> ",mat)

#Feature 13
l = [item for sublist in stemmed_list for item in sublist]
s = Counter(l)
sol = []
for i in range(len(stemmed_list)):
	d = Counter(stemmed_list[i])
	t = 0;
	for j in range(len(stemmed_list[i])):
		p = s.get(stemmed_list[i][j])
		q = d.get(stemmed_list[i][j])
		t = t + (p*q)
	sol.append(t)
#print("\nTotal tl * tf --> ", sol)

#Feature 14
#print("\nCosine similarity b/w segment & whole document --> ", cos_seg_doc)

#Feature 15
rs = [item for sublist in stemmed_list for item in sublist]
sp = Counter(rs)
res = []
for i in range(len(stemmed_list)):
   t = 0
   for j in range(len(stemmed_list[i])):
      p = sp.get(stemmed_list[i][j])
      if p > 1:
         t = t + 1
   td = t / len(stemmed_list)
   res.append(td)
#print("\nLexical analysis --> ", res)

#Feature 16
is_uppercase = []
for i in range(len(in_words)):
	w = [word for word in in_words[i] if word.isupper()]
	if len(w) == 0:
		is_uppercase.append(0)
	else:
		is_uppercase.append(1)
#print("\nDoes segment contain uppercase word --> ", is_uppercase)

#Feature 17
is_pronoun = []
for i in range(len(in_sent)):
	for j in range(len(in_sent[i])):
		sentence = in_sent[i][j]
		tagged_sent = pos_tag(sentence.split())
		propernouns = [word for word,pos in tagged_sent if pos == 'PRP']
		if len(propernouns) == 0:
			is_pronoun.append(0)
		else:
			is_pronoun.append(1)
#print("\nDoes segment contain a pronoun --> ", is_pronoun)

#Feature 18
is_propernoun = []
for i in range(len(in_sent)):
	for j in range(len(in_sent[i])):
		sentence = in_sent[i][j]
		tagged_sent = pos_tag(sentence.split())
		propernouns = [word for word,pos in tagged_sent if pos == 'NNP']
		if len(propernouns) == 0:
			is_propernoun.append(0)
		else:
			is_propernoun.append(1)
#print("\nDoes segment contain a proper noun", is_propernoun)

#Feature 19
counts = Counter(stemmed_words)
mc_list = []
m_list = []
p = list(counts.values())
q = len(p) - p.count(1)
#print("\n",)
for word, count in counts.most_common(q):
   mc_list.append(word)
Tot_sig_terms = len(mc_list)
sig_term = []
significant_terms = []
for i in range(len(stemmed_list)):
   v1 = list(set(stemmed_list[i]) & set(mc_list))
   significant_terms.append(v1)
   v2 = len(v1)/Tot_sig_terms
   sig_term.append(v2)
#print("\nSignificant terms -->", sig_term)

 
#Feature 20
merged_significant_terms = list(itertools.chain(*significant_terms))
z_val = Counter(merged_significant_terms)
sim_seg_sig = cosine(stemmed_list,z_val)
#print("\nCosine similarity b/w segment & significant terms -->", sim_seg_sig)

#Feature 21
sig_term_luhn = []
for i in range(len(stemmed_list)):
   st = ((len(significant_terms[i]))**2)/(len(stemmed_list[i]))
   sig_term_luhn.append(st)
#print("\nSignificant term factor calculated according to luhn --> ", sig_term_luhn)

#Feature 22
mod_luhn_factor = []
for i in range(len(stemmed_list)):
   t = sig_term_luhn[i] + title_word[i]
   mod_luhn_factor.append(t)
#print("\nModified Luhn's significant term factor --> ",mod_luhn_factor)

#************************ Feature Generation End ******************************

f_matrix = numpy.column_stack((seg_id,para_id,para_off,para_loc,seg_loc,seg_len,title_word,cos_seg_title,total_tf,average_tf,sum_isf,mat,sol,cos_seg_doc,res,is_uppercase,is_pronoun,is_propernoun,sig_term,sim_seg_sig,sig_term_luhn,mod_luhn_factor))
#print("\nFeature Matrix",f_matrix)              #Feature Matrix


#********* Sending Feature Matrix to CSV **********


numpy.savetxt("Feature Matrix.csv", f_matrix, delimiter=",",fmt='%10.5f')


#**********   End **********************************


U, s, V = SL.svd(f_matrix, full_matrices=False)  #Singular Valued Decomposition

k_value = []
for i in range(k_input):
   k_value.append(i)

extract = U[:,k_value]   #Feature matrix obtained after applying SVD on previous Feature Matrix and Selecting k columns.

numpy.savetxt("Extract.csv", extract, delimiter=",",fmt='%10.5f')

shape_extracted = extract.shape
#print("\nExtracted matrix-->",extract)
#print("\nShape -->",shape_extracted)

similarity = cosine_similarity(extract)  #Similarity matrix containing similarity b/w each pair nodes.
shape_similarity = similarity.shape
#print("\nExtracted matrix-->",similarity)
#print("\nShape -->",shape_similarity)

#********* Sending Similarity Matrix to CSV **********

f_name = "Similarity matrix for k = " + str(k_input) +".csv"
numpy.savetxt(f_name, similarity, delimiter=",",fmt='%10.5f')


#**********   End **********************************
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



#********* Sending Similarity Matrix which we get after taking threshold to CSV **********

f1_name = "Graphs Similarity matrix for k = " + str(k_input) +".csv"
numpy.savetxt(f1_name, similarity, delimiter=",",fmt='%10.5f')


#**********   End **********************************
     


G=nx.from_numpy_matrix(similarity,create_using=nx.DiGraph())   #Graph object created from similarity matrix.

##################################################################################
'''
pos = nx.spring_layout(G,k=1,iterations=60)
labels = {i : i + 1 for i in G.nodes()}
nx.draw_networkx_labels(G, pos, labels, font_size=10)
plt.savefig("simple_path3.png")
'''
##################################################################################



pr = nx.pagerank(G, alpha=0.15) #Finds the PageRank of Graph

st = Counter(pr)
op = []
for k,v in st.most_common(int(dimen/5)):  #Compression Rate 20%,i.e, (no. of sentences/5)
   op.append(k)
print("\nSentences --> ",dimen)
print("Order--> ",op)
'''
for i in range(len(op)):
   print("\n***",op[i],"--> ",sentence_form[op[i]])
'''
sent_in_para = []
for i in range(len(op)):
   k = find_para(sentence_form[op[i]],in_sent)
   sent_in_para.append(k)

print("\nParagraph locations are::",sent_in_para)
   
f2 = "Summary for k = " + str(k_input) +".txt"

with open(f2, 'a') as f:
   print('\nSummary when k and similarity threshold is taken as ->',k_input,' and ',similarity_threshold,file=f)
   if type_of_graph == 'UD':
      print('And type of graph is Undirected.',file=f)
   if type_of_graph == 'FD':
      print('And type of graph is Forward directed.',file=f)
   if type_of_graph == 'BD':
      print('And type of graph is Backward directed.',file=f)
      
   print('\nSummary is-->',file=f)
   for i in range(len(op)):
      print(op[i]+1,".",sentence_form[op[i]],file=f)

with open("Sentences.txt",'a') as f:
   print("Sentence_form -> ",sentence_form,file=f)

input("\nPress any key to exit.")
