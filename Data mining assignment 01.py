
#import all libiraries
import os
import math
import nltk
from nltk.tokenize import RegexpTokenizer
from collections import defaultdict
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

#Define_text_term function to convert text into term using tokenizer,stopword and stemmer
def text_to_term(doc):
    tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
    #Tokenization
    tokens = tokenizer.tokenize(doc)    
    #stemmer function
    stemmer = PorterStemmer()
    #Stop words removal
    tokens_new = [token for token in tokens if token not in stopwords.words('english')]  
    #Stemming the tokens
    stem_token = [stemmer.stem(token) for token in tokens_new] 
    return stem_token

#get_idf fucntion, return idf value of a give token otherwise return -1
def getidf(token):    
    if doc_freq[token]!= 0:
        return math.log10((len(term_freq) / doc_freq[token]))
    return -1

#Computing TF-IDF score from term_freq and docu_freq table and normalize it
def tfidf_vector(term_frequency):
    for doc in term_frequency:
        tf= term_frequency[doc]
        sum_weight=0
        tfidf_norm[doc] = defaultdict(int)
        for token in tf:
            idf_value = getidf(token)
            tfidf_result = (1 + math.log10(tf[token])) * idf_value
            tfidf_norm[doc][token] = tfidf_result
            sum_weight += (tfidf_result *tfidf_result)
        sum_average[doc] = math.sqrt(sum_weight)       
    for doc in tfidf_norm:
        for token in tfidf_norm[doc]:
            if sum_average[doc]:
                tfidf_norm[doc][token] =tfidf_norm[doc][token]/ sum_average[doc]
            if token not in dictionary_tf_idf :
                dictionary_tf_idf [token] = Counter()
            dictionary_tf_idf [token][doc] = tfidf_norm[doc][token]
            

#getweight function return normalize_tf_idf value, otherwise return 0
def getweight(filename,token):
    #stem the given token
    tokens = stemmer.stem(token)
    if tokens in tfidf_norm[filename]:
        return tfidf_norm[filename][tokens]
    return 0

#query function return cosine similarity score of a given docuemnt and a query using lnc.ltc weight
def query(qstring):
    q_token = text_to_term(qstring)
    q_token_freq = Counter(q_token)
    query_tf = {}
    count = {}
    q_file,query_sum = 0,0
    matchDoc = {}
    for q_term in q_token:
        if q_term in dictionary_tf_idf:
                matchDoc[q_term], sum_token = zip(*dictionary_tf_idf[q_term].most_common(10))                
        if q_file == 0:
            doc_token_match = set(matchDoc[q_term]) #gives docs where token is matching
            q_file = 1
        else:
            doc_token_match = set(matchDoc[q_term]) & doc_token_match
        if len(sum_token) < 10:
            count[q_term] = 0
        else:
            count[q_term] = sum_token[9]
            
        query_tf[q_term] = 1+math.log10(q_token_freq[q_term])    
        query_sum+= math.pow(query_tf[q_term], 2)
    query_sum = math.sqrt(query_sum)  
    
    # matched_doc, cosine_sim_score = query_vector(matchDoc,count, query_sum, query_tf) 
    match = Counter()
    for doc in tfidf_norm:
        matched_value = 0
        for token in query_tf:
            if len(count) != 0 and doc not in matchDoc[token]:
                matched_value += (query_tf[token] / query_sum) * count[token]
            if doc in matchDoc[token]:
                matched_value += (query_tf[token] / query_sum) * dictionary_tf_idf[token][doc]
        match[doc] =matched_value
    for i in match.most_common(1): 
        max_simi_doc= i[0]
        max_value = i[1]
    return max_simi_doc, max_value



#global_variable for term and document frequecny
#term_frequency dictionary
term_freq = {}
#docuemnt frequency dictionary
doc_freq = defaultdict(int)
tfidf_norm = {}
sum_average = defaultdict(int)
#normalize tf_idf table
dictionary_tf_idf = defaultdict(lambda: defaultdict(int))
#matching document dictionary
matchDoc = {}
stemmer = PorterStemmer()

#main function
def main():
    tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
    stemmer = PorterStemmer()
    folder_path = './US_Inaugural_Addresses'   
    for filename in os.listdir(folder_path):
        if filename.startswith('0') or filename.startswith('1'):
            file = open(os.path.join(folder_path, filename), "r", encoding='windows-1252')
            doc = file.read()
            file.close() 
            doc = doc.lower()
            tokens = text_to_term(doc)
            for token in set(tokens):
                doc_freq[token] += 1
            counter_tf = defaultdict(int)
            for token in tokens:
                counter_tf[token] += 1
            term_freq[filename] = counter_tf.copy()
            counter_tf.clear()            
    tfidf_vector(term_freq)
    #print fucntion
    print("%.12f" % getidf('british'))
    print("%.12f" % getidf('union'))
    print("%.12f" % getidf('war'))
    print("%.12f" % getidf('military'))
    print("%.12f" % getidf('great'))
    print("--------------")
    print("%.12f" % getweight('02_washington_1793.txt','arrive'))
    print("%.12f" % getweight('07_madison_1813.txt','war'))
    print("%.12f" % getweight('12_jackson_1833.txt','union'))
    print("%.12f" % getweight('09_monroe_1821.txt','british'))
    print("%.12f" % getweight('05_jefferson_1805.txt','public'))
    print("--------------")
    print("(%s, %.12f)" % query("pleasing people"))
    print("(%s, %.12f)" % query("british war"))
    print("(%s, %.12f)" % query("false public"))
    print("(%s, %.12f)" % query("people institutions"))
    print("(%s, %.12f)" % query("violated willingly"))
if __name__ == "__main__":
    main()


