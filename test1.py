import gensim
from weightCodeJi import TFKLD
from pickle import load, dump
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from scipy import spatial
import numpy
import nltk
import test2
import gzip

# Python program to translate
# speech to text and text to speech
import speech_recognition as sr
import pyttsx3
import webbrowser as wb
from hindi import *
# Initialize the recognizer
r = sr.Recognizer()
# functin to speak
def SpeakText(command,i):
    	
	# Initialize the engine
	engine = pyttsx3.init()
	voices = engine.getProperty('voices')
	engine.setProperty("voice", voices[i].id)
	engine.say(command)
	engine.runAndWait()
	
'''
engine = pyttsx3.init()
voices = engine.getProperty('voices')
for voice in voices:
    print(f"Voice: {voice.name}")

'''
def writeInFile(S,mode):
	file1 = open("t.data", mode)
	file1.writelines(S)
	file1.close()

Questions = ["What was President Donald Trump's prediction?", "What is the number of confirmed cases in US?"]
Context =  "The US has passed the peak on new coronavirus cases, " \
            "President Donald Trump said and predicted that some states would reopen this month. " \
            
def takeCommandEnglish():
	try:		
	    # use the microphone as source for input.
		with sr.Microphone() as source2:			
			# wait for a second to let the recognizer
			# adjust the energy threshold based on
			# the surrounding noise level
			r.adjust_for_ambient_noise(source2, duration=1)
			print('Give first sentence')
			
			#listens for the user's input
			audio1 = r.listen(source2)
			
			# Using ggogle to recognize audio
			s1 = r.recognize_google(audio1)
			# s1 = r.recognize_bing(audio2)
			# s1 = r.recognize_ibm(audio2)
			# s1 = s1.lower()
			writeInFile(s1+"\t","w")
            
			r.adjust_for_ambient_noise(source2, duration=2)
			print('Give second sentence')
			audio2 = r.listen(source2)
			s2 = r.recognize_google(audio2)
			writeInFile(s2,"a")
			
			# print("Did you say "+ MyText)
			# print('It has been opened in chrome')

			# SpeakText("Hey Sandeep, did you just said " + MyText,2)
			
	except sr.RequestError as e:
		print("Could not request results; {0}".format(e))
		
	except sr.UnknownValueError:
		print("unknown error occured")
    

# explicit function to take input commands
# and recognize them

# call the function
print("Type 1 for Hindi or 2 for English")
i = int(input())
if i ==1:
    takeCommandHindi()
else:
	takeCommandEnglish()


print("Wait for the result")
SpeakText("Wait for the result ",0)

# Load Google's pre-trained Word2Vec model.
#f_in = gzip.open('GoogleNews-vectors-negative300.bin.gz', 'rb')
#f_out = open('GoogleNews-vectors-negative300.bin', 'wb')
#f_out.writelines(f_in)
#model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True, limit=100000)

model = load(open("word2vec.pickle", 'rb'))


# print("Before weights load")
with open('weightsTempAll.pickle', 'rb') as handle:
            weights = load(handle)


# print( "After Weights Load")
def getSentenceList(fname):
    text, label = [], []
    i = 0
    
    with open(fname, 'r', encoding="utf-8-sig") as fin:
        lines = list(fin)
    
    for line in lines:
        
        #print lines[i]
        # print( lines[i].split("\t")[1])
        #print lines[i].split("\t")[2]
        text.append(line.split("\t")[0])
        text.append(line.split("\t")[1])
        #print "Pair number", i, "parsed"
        i += 1
    
    return text

avgTfkld = 0
for word in weights:
    avgTfkld = avgTfkld + weights[word]

avgTfkld = avgTfkld/len(weights)

def getVectors(f):
    vectors = []
    bleu = []
    lengthDiff = []
    i = 0
    text = getSentenceList(f)
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
    return combinedVecs


vecs = getVectors("t.data")
classifier = load(open('model.pickle', 'rb'))
result_val = classifier.predict(vecs)
print("score is",result_val)
if result_val[0] == 1:
    SpeakText("Both sentence have similar meaning ",0)
else:
    SpeakText("Both sentence does not have similar meaning ",0)