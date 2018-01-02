from find_para import *
import numpy
from nltk import word_tokenize, sent_tokenize, pos_tag
text = open('test.txt').read()
in_para = text.split('.\n')
in_sent = []
for i in range(len(in_para)):
    sent = sent_tokenize(in_para[i]) #Sentence tokenizer
    in_sent.append(sent)
sentence_form = sent_tokenize(text)
#x = find_para(sentence_form[1],in_sent)
c = sentence_form[1]
x = [(i, colour.index(c))
 for i, colour in enumerate(in_sent)
 if c in colour]

print("A--->",x,in_sent[1],sentence_form[1])
