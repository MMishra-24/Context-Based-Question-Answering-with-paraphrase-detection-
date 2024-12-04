import gensim
from weightCodeJi import TFKLD
from pickle import load, dump
import pickle
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from scipy import spatial
import numpy
import nltk

import gzip
# Load Google's pre-trained Word2Vec model.
f_in = gzip.open('GoogleNews-vectors-negative300.bin.gz', 'rb')
f_out = open('GoogleNews-vectors-negative300.bin', 'wb')
f_out.writelines(f_in)
model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True, limit=100000)
"""sentence = ["London", "is", "the", "capital", "Great", "Britain"]
vectors = [model[w] for w in sentence]
print vectors

ftrain = "train.data"
fdev = "dev.data"
ftest = "test.data"
tfkld = TFKLD(ftrain, fdev, ftest)
tfkld.weighting()
tfkld.createdata()
"""
print("Before weights load")
with open('weightsTempAll.pickle', 'rb') as handle:
            weights = load(handle)

"""
tk = TFKLD("train.data", "sample.data", "test.data")
tk.weighting()
weights = tk.weightVocab"""
print( "After Weights Load")
def getSentenceList(fname):
    text, label = [], []
    i = 0
    sentenceIdRemover = re.compile('\\t\d+\\t\d+\\t')
    utfBOMRemover = re.compile('\\xef\\xbb\\xbf')
    with open(fname, 'r', encoding="utf-8-sig") as fin:
        lines = list(fin)
    
    for line in lines:
        line = sentenceIdRemover.sub("\t", line)
        line = utfBOMRemover.sub(" ", line)
        #print lines[i]
        # print( lines[i].split("\t")[1])
        #print lines[i].split("\t")[2]
        text.append(line.split("\t")[1])
        text.append(line.split("\t")[2])
        label.append(int(line.split("\t")[0]))
        #print "Pair number", i, "parsed"
        i += 1
    return text, label

avgTfkld = 0
for word in weights:
    avgTfkld = avgTfkld + weights[word]

avgTfkld = avgTfkld/len(weights)
"""
print weights

sam = ["This", "is", "a", "string"]
for word in sam:
    if word in model and word in weights:
        print word
print "partition"

for word in sam:
    if word in weights:
        print word
"""


def getVectors(f):
    vectors = []
    bleu = []
    lengthDiff = []
    i = 0
    text, label = getSentenceList(f)
    for sentenceNumber in range(len(text)):
        sentence = text[sentenceNumber]
        

        wordList = re.compile('\w+').findall(sentence)
        if sentenceNumber % 2 == 0:
            # print len(text), sentenceNumber + 1
            # print(len(text), sentenceNumber + 1)
            hypothesis = text[sentenceNumber + 1]
            hypSentenceList = re.compile('\w+').findall(hypothesis)
            lengthDiff.append(abs(len(wordList) - len(hypSentenceList)))
            bleu.append(nltk.translate.bleu_score.sentence_bleu([wordList], hypSentenceList))
        weightedSum = [0] * 300
        for word in wordList:
            #print word
            if word in model and word in weights:
                weightedSum = weightedSum + model[word] * weights[word]
            
            elif word in model and word not in weights:
                weightedSum = weightedSum + (model[word] * avgTfkld)

            else:
                pass
                #print model[word]
                #print "in both"
                #print vector
        vectors.append(weightedSum)
        #print i, "Sentence converted"
        i+=1

    combinedVecs = []
    for i in range(0, len(vectors), 2):
        temp = numpy.append(vectors[i], vectors[i+1])
        temp2 = numpy.append(temp, spatial.distance.euclidean(vectors[i], vectors[i + 1]))
        temp3 = numpy.append(temp2, bleu[int(i/2)])
        temp4 = numpy.append(temp3, lengthDiff[int(i/2)])
        combinedVecs.append(temp3)
    return combinedVecs, label

#vecs, label = getVectors("train2.data")
vecs, label = getVectors("train.data")
#svclassifier = LinearSVC(random_state = 0, C = 0.20, penalty = "l2", loss = "l1")

#svclassifier.fit(vecs, label)
#print "classifier coefficients", (svclassifier.coef_)

#testVecs, testLabels = getVectors("test2.data")

testVecs, testLabels = getVectors("test.data")
#score = svclassifier.score(testVecs, testLabels)
classifier = LogisticRegression(random_state = 0, C = 0.20, penalty = "l2")
classifier.fit(vecs, label)
# classifier.predict
score = classifier.score(testVecs, testLabels)
pickle.dump(classifier, open('model.pickle', 'wb'))

print("score is",score)