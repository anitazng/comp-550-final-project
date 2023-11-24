from pypinyin import pinyin, lazy_pinyin, Style
import numpy as np
from gensim.models import Word2Vec
import csv

#returns pinyin
def pychar(char):
    return lazy_pinyin(char, style=Style.TONE3)[0]

#frequency of character in list
def freq(char, lst):
    count = 0
    for row in lst:
        for l in row:
            if char == l:
                count += 1
    return count

#creates list of the characters
def charLst(lst):
    charList = []
    for row in lst:
        for char in row:
            if char not in charList:
                charList.append(char)
    return charList

#finds a character of similar frequency for baseline
def simFreq(char, lst):
    frequency = freq(char, lst)
    for compare in charLst(lst):
        if freq(compare, lst) == frequency:
            return compare
    for i in range[1:100]:
        if (freq(compare, lst) + i == frequency) or (freq(compare, lst) - i == frequency):
            return compare
        
#calculates homophone distance
def homoDist(homophones, model):
    baseline = homophones[0]
    compare = simFreq(baseline, lst)
    baseCompare = model.wv.similarity(baseline, compare)
    homoCompare = []
    for i in range(1,len(homophones)):
        homoCompare.append(model.wv.similarity(baseline, homophones[i]))
    # you can use this to see the homophones being compared, the chosen  homophone (automatically chosen to be the first in the list), the same-frequency character used as baseline, and the distance between the chosen homophone and the other homophones in the list
    # print('HOMOPHONE BEING COMPARED:' + homophones[0])
    # print('comparison with baseline: ' + compare)
    # print(baseCompare)
    # for i in range(0,len(homoCompare)):
    #     print('comparison with homophone: ' + homophones[i+1])
    #     print(homoCompare[i])
    return baseCompare - homoCompare[0]

lst  = []

with open('../transcripts.tsv', newline ='', mode="r", encoding='utf-8') as file:
    reader = csv.reader(file)
    for row in reader:
        lst.append(str(row))

chardict = {}

for row in lst:
    for char in list(row):
        if pychar(char) in chardict:
            if char not in "[]'":
                if char not in chardict[pychar(char)]:
                    chardict[pychar(char)].append(char)
        else:
            chardict[pychar(char)] = [char]

            
for row in chardict:
    if len(chardict[row]) > 1:
        print(row)
        print(chardict[row])

'''word2vec stuff'''

#this part is just to get a list of all the sentences

lst = []
with open('../transcripts.tsv', newline ='', mode="r", encoding='utf-8') as file:
    reader = csv.reader(file)
    for row in reader:
        lst.append(str(row))

mlst = []

for l in lst:
    #get rid of the enclosing punctuation
    mstr = l[2:]
    mstr = mstr[:(len(l)-4)]
    mlst.append(mstr)

#creates list
sentences = []
for l in mlst:
    sentences.append(list(l))

#generates word2vec model
model = Word2Vec(sentences, vector_size=300, min_count = 1)

# finds average of similarity differences

# sims = []
# for i in chardict.keys():
#     try:
#         sims.append(homoDist(chardict[i], model))
#     except: KeyError

# print(np.mean(sims))