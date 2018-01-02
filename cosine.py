from collections import Counter
def cosine(list1,val):
    cosine = [];
    for i in range(len(list1)):
        a_val = Counter(list1[i])
        words  = list(a_val.keys() | val.keys())
        a_vect = [a_val.get(word, 0) for word in words]
        b_vect = [val.get(word, 0) for word in words]
        len_a  = sum((av*av) for av in a_vect) ** (0.5)
        len_b  = sum((bv*bv) for bv in b_vect) ** (0.5)
        dot    = sum((av*bv) for av,bv in zip(a_vect, b_vect))
        cosi = dot / ((len_a * len_b)) 
        cosine.append(cosi)
    return cosine
    
