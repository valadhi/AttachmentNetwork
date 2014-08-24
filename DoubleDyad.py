import restrictedBoltzmannMachine as rbm
import numpy as np
import cPickle as pickle
import readKanade
import matplotlib.pyplot as plt
from common import *
import os, argparse
from activationfunctions import *
from sklearn import linear_model#,metrics
import time
#import cv2
import matplotlib.image as io
import copy

from sklearn.cross_validation import train_test_split
from sklearn import linear_model, datasets, metrics
start_time = time.time()

small_size = (25,25)
large_size = (50,50)
size = large_size

def buildParent(inputEmotions): # trains network with emotional associations	

	#asoc = {"fear": "happy", "anger":"fear", "happy":"anger"}
	#childLabelList = {"happy":None, "anger":None, "fear":None}
	data,labels,childLabelList = readKanade.readProportion(inputEmotions)
	print "data.shape"
	print data.shape
	print "labels.shape",labels.shape
	#print "labellist ", childLabelList["happy"]

	activationFunction = Sigmoid()

	Data = np.concatenate((data, labels), axis=1)
	np.random.shuffle(Data)
	finalTrainingData = Data[0:-1, :]

	nrVisible = len(finalTrainingData[0])
	nrHidden = 800


	parentNet = rbm.RBM(nrVisible, nrHidden, 0.01, 0.8, 0.8,
					visibleActivationFunction=activationFunction,
					hiddenActivationFunction=activationFunction,
					rmsprop=True,#args.rbmrmsprop,
					nesterov=True,#args.rbmnesterov,
					sparsityConstraint=False,#args.sparsity,
					sparsityRegularization=0.5,
					trainingEpochs=10,#args.maxEpochs,
					sparsityTraget=0.01,
					fixedLabel = True)

	parentNet.train(finalTrainingData)
	t = visualizeWeights(parentNet.weights.T, (size[0]*2,size[1]), (10,10))
	plt.imshow(t, cmap=plt.cm.gray)
	plt.axis('off')
	plt.savefig('dump/weightsparentNet.png', transparent=True)

	return parentNet, childLabelList

# returns emotion dictionary of {"emotion" : ndarray of resulting image}
def interactChild(pNet1, pNet2, childDataSetOfEmotions, parentPercentages):
	outputEmotions = copy.copy(childDataSetOfEmotions)
	# 1. feed the child emotions into parent
	sizeOfParentFeedback = 20
	rando = np.random.random_sample(size)
	rando = rando.reshape(1, size[0]**2)
	#saveImage(rando, "rando_",(25,25), "parentchildoutput")	
	#print rando
	parentResponses = np.array([])
	
	'''
	# generate sample images as returned from parent childNet for each child emotion
	for key in childDataSetOfEmotions.iterkeys():
		emotion = childDataSetOfEmotions[key]

		saveImage(emotion, key+"EMOTION1",(size[0],size[1]), "testemo")

		emotion = emotion.reshape(1, emotion.shape[0])
		recon = np.concatenate((rando, emotion), axis=1)
		recon = pNet1.reconstruct(recon,3)
		saveImage(recon, key,(size[0]*2,size[1]), "parentchildoutput")
		outputEmotions[key] = recon[:,:size[0]**2]
	'''
	for key in childDataSetOfEmotions.iterkeys():
		for s in xrange(parentPercentages[0]*sizeOfParentFeedback):
			emotion = childDataSetOfEmotions[key]
			#print "emotion ",emotion
			#print "emotion ",emotion.shape
			saveImage(emotion, key+"EMOTION2",(size[0],size[1]), "testemo")
			#if emotion.shape[0] != 1:
			#print "EMOTION SHAPE",emotion.shape
			emotion = emotion.reshape(1, emotion.shape[0])
			#print "EMOTION SHAPE",emotion.shape
			recon = np.concatenate((rando, emotion), axis=1)
			#print "recon shape ",recon.shape
			#print "type ",recon[0,1],type(recon[0,1])
			recon = pNet1.reconstruct(recon,300)
			if parentResponses.size == 0:
				parentResponses = recon
				#saveImage(recon, key+"anexample",(size[0]*2,size[1]), "parentchildoutput")
			else:
				parentResponses = np.vstack((parentResponses, recon))
				#saveImage(recon, key+"anexample"+str(s),(size[0]*2,size[1]), "parentchildoutput")

	for key in childDataSetOfEmotions.iterkeys():
		for s in xrange(parentPercentages[1]*sizeOfParentFeedback):
			emotion = childDataSetOfEmotions[key]
			saveImage(emotion, key+"EMOTION2",(size[0],size[1]), "testemo")
			emotion = emotion.reshape(1, emotion.shape[0])
			recon = np.concatenate((rando, emotion), axis=1)
			recon = pNet2.reconstruct(recon,300)
			if parentResponses.size == 0:
				parentResponses = recon
			else:
				parentResponses = np.vstack((parentResponses, recon))		


	print "parentShape ",parentResponses.shape

	# 2. train the child on the outputs from the parent	
	nrVisible = len(parentResponses[0])
	nrHidden = 600
	activationFunction = Sigmoid()

	print "visible1 ",nrVisible
	print "data row1 ",parentResponses[0,:].shape 

	childNet = rbm.RBM(nrVisible, nrHidden, 0.01, 0.8, 0.8,
					visibleActivationFunction=activationFunction,
					hiddenActivationFunction=activationFunction,
					rmsprop=True,#args.rbmrmsprop,
					nesterov=True,#args.rbmnesterov,
					sparsityConstraint=False,#args.sparsity,
					sparsityRegularization=0.5,
					trainingEpochs=70,#args.maxEpochs,
					sparsityTraget=0.01,
					fixedLabel = True)

	childNet.train(parentResponses)

	t = visualizeWeights(childNet.weights.T, (size[0]*2,size[1]), (10,10))
	plt.imshow(t, cmap=plt.cm.gray)
	plt.axis('off')
	plt.savefig('dump/weightschildNet.png', transparent=True)

	# generate emotions from network trained on parent emotional feedback
	for key in childDataSetOfEmotions.iterkeys():
		print "child Emotion ",key
		emotion = childDataSetOfEmotions[key]
		saveImage(emotion, key+"EMOTION3",(size[0],size[1]), "testemo")
		if emotion.shape[0] != 1:
			emotion = emotion.reshape(1, emotion.shape[0])
		recon = np.concatenate((rando, emotion), axis=1)
		print "recon shape ",recon.shape
		recon = childNet.reconstruct(recon,3)
		outputEmotions[key] = recon[:,:size[0]**2]# first half is reconstructed bit
		saveImage(recon, key+"afterParent1",(size[0]*2,size[1]), "parentchildoutput")
	
	# 3. classify the resulting emotion
	return outputEmotions
		
def scikitclassifier():
	X, Y = readKanade.readAllEmotionssk()
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
	                                                    test_size=0.19,
	                                                    random_state=0)


	print("X_train",X_train.shape)
	print("X_test",  X_test.shape)
	print("Y_train", Y_train.shape)
	print("Y_test", Y_test.shape)
	# Training Logistic regression
	logistic_classifier = linear_model.LogisticRegression(C=100.0)
	logistic_classifier.fit(X_train, Y_train)

	print("Logistic regression using raw pixel features:\n%s\n" % (
    metrics.classification_report(
        Y_test,
        logistic_classifier.predict(X_test))))

	return logistic_classifier

def emoEval(classifier, emotions):
	emotionsdct = {1:"fear", 2:"happy",3:"anger",4:"contempt",5:"disgust",6:"sadness",7:"surprise"}
	results = {}
	for emote in emotions.iterkeys():	
		emotion = emotions[emote]
		saveImage(emotion, emote+"FINAL",(50, 50), "parentchildoutput")

		#print "Emotion: ",emotionsdct[classifier.predict(emotion)]
		prediction = classifier.predict(emotion)
		print "Expected ",emote," Classified as:",emotionsdct[prediction[0]]

		results[emote] = emotionsdct[prediction[0]]


	return results			

def saveImage(data, name, size,temp=""):
	#print "data ",data.shape
	#print "size ",size
	plt.imshow(vectorToImage(data, size), cmap=plt.cm.gray)
	plt.axis('off')
	#data = cv2.resize(data, size)
	if temp == "":
		plt.savefig("dump/"+name + '.png',transparent=True)
		#io.imsave("dump/"+name + '.png', data, cmap=plt.cm.gray)
	else:
		if(not os.path.exists("dump/"+temp+"/")):
			os.makedirs("dump/"+temp+"/")#
		plt.savefig("dump/"+ temp + "/"+name + '.png',transparent=True)
		#io.imsave("dump/"+ temp + "/"+name + '.png', data, cmap=plt.cm.gray) 

def run(parent1, parent2, childEmotionalProportions, parentPercentages):


	pNet1,childLabels1 = buildParent(parent1)
	pNet2,childLabels2 = buildParent(parent2)
	
	for key in childLabels1.iterkeys():
		emotion = childLabels1[key]
		saveImage(emotion, key+"LABELS1",size, "parentchildoutput")
	for key in childLabels2.iterkeys():
		emotion = childLabels2[key]
		saveImage(emotion, key+"LABELS2",size, "parentchildoutput")
	
	emotionResponses = interactChild(pNet1, pnet1, childLabels1, parentPercentages)
	#emoClassifier, emoLabels = trainEmotionClassifier()

	outputs = emoEval(scikitclassifier(), emotionResponses)

	#runEmoEval(emoClassifier, emoLabels, emotionResponses)
	print "exited"
	print("--- %s seconds ---" % (time.time() - start_time))

	return outputs
'''
def main():
	scikitclassifier()

if __name__ == '__main__':
	main()
'''
