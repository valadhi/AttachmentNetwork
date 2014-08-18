import restrictedBoltzmannMachine as rbm
import numpy as np
import cPickle as pickle
import readKanade
import matplotlib.pyplot as plt
from common import *
import os, argparse
from activationfunctions import *
from sklearn import linear_model#,metrics

parser = argparse.ArgumentParser(description='dyad simulation')
parser.add_argument('--trainParent',dest="trainParent", help="train and save Parent net"
	, default = False )
parser.add_argument('--trainChild',dest="trainChild", 
	help="generate training data from parent and train child net", default = False)
parser.add_argument('--loadParent',dest='loadParent', default=False,
					help="if true, the parent network is loaded")
parser.add_argument('--loadChild',dest='loadChild', default=False,
					help="if true, the child network is loaded")
#parser.add_argument('--classify',dest='classify',action='store_true', default=False,
#					help="if true, the child network is loaded")

args = parser.parse_args()
argsdict = vars(args)

for k in argsdict.iterkeys():
	print k, type(k)
	print argsdict[k]
	print "#####"
small_size = (25,25)
large_size = (50,50)
size = large_size

def buildParent(inputEmotions): # trains network with emotional associations	
	if argsdict["loadParent"] != False:
		print "in load parentNet"
		f = open(argsdict["loadParent"], "rb")
		parentNet = pickle.load(f)
		childLabelList = pickle.load(f)
		f.close()
		print "out of load parentNet"
	else:
		#asoc = {"fear": "happy", "anger":"fear", "happy":"anger"}
		#childLabelList = {"happy":None, "anger":None, "fear":None}
		data,labels,childLabelList = readKanade.readProportion(inputEmotions)
		print "data.shape"
		print data.shape
		print "labels.shape",labels.shape

		print type(data)
		print type(labels)
		#data = data / 255.0
		#labels = labels / 255.0
		'''
		print "before trial"
		for k in childLabelList.iterkeys():
			print k
			saveImage(childLabelList[k], k, size, "labels")
		for i in xrange(len(data)):
			saveImage(data[i], str(i)+"data", size, "data")
			saveImage(labels[i], str(i)+"label", size, "labels")
		print "after trial"
		'''
		activationFunction = Sigmoid()

		Data = np.concatenate((data, labels), axis=1)
		#np.random.shuffle(Data)
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
						trainingEpochs=15,#args.maxEpochs,
						sparsityTraget=0.01,
						fixedLabel = True)

		parentNet.train(finalTrainingData)
		t = visualizeWeights(parentNet.weights.T, (size[0]*2,size[1]), (10,10))
		plt.imshow(t, cmap=plt.cm.gray)
		plt.axis('off')
		plt.savefig('dump/weights.png', transparent=True)

	if argsdict["trainParent"] != False:
		with open(argsdict["trainParent"], "wb") as f:
			pickle.dump(parentNet, f)
			pickle.dump(childLabelList, f)
	return parentNet, childLabelList

def saveImageFoo(data, name, size):
  plt.imshow(vectorToImage(data, size), cmap=plt.cm.gray)
  plt.axis('off')
  plt.savefig("dump/"+name + '.png',transparent=True) 

# returns emotion dictionary of {"emotion" : ndarray of resulting image}
def interactChild(parentNet, childDataSetOfEmotions):
	outputEmotions = childDataSetOfEmotions
	# 1. feed the child emotions into parent
	sizeOfParentFeedback = 40
	rando = np.random.random_sample(size)
	rando = rando.reshape(1, size[0]**2)
	#saveImage(rando, "rando_",(25,25), "parentchildoutput")	
	#print rando
	parentResponses = np.array([])
	'''
	# generate sample images as returned from parent childNet for each child emotion
	for key in childDataSetOfEmotions.iterkeys():
		emotion = childDataSetOfEmotions[key]
		emotion = emotion.reshape(1, emotion.shape[0])
		recon = np.concatenate((rando, emotion), axis=1)
		recon = parentNet.reconstruct(recon,300)
		saveImage(recon, key,(size[0]*2,size[1]), "parentchildoutput")	
	'''
	print "enter interact"

	if argsdict["loadChild"] != False: # load previously generated parent emotional feedback database
		f = open(argsdict["loadChild"], "rb")
		childNet = pickle.load(f)
		#parentResponses = pickle.load(f)
		f.close()
	else:# generate new parent emotional feedback database and train child
		for key in childDataSetOfEmotions.iterkeys():
			for s in xrange(sizeOfParentFeedback):
				emotion = childDataSetOfEmotions[key]
				print "emotion ",emotion
				emotion = emotion.reshape(1, emotion.shape[0])
				recon = np.concatenate((rando, emotion), axis=1)
				print "recon shape ",recon.shape
				print "type ",recon[0,1],type(recon[0,1])
				recon = parentNet.reconstruct(recon,30)
				if parentResponses.size == 0:
					parentResponses = recon
				else:
					parentResponses = np.vstack((parentResponses, recon))
				#saveImage(recon, key,(size[0]*2,size[1]), "parentchildoutput")

	
		print "parentShape ",parentResponses.shape

		'''
		print "before saving all parent emotional feedback"
		for k in xrange(parentResponses.shape[0]):
			saveImage(parentResponses[k,:], str(k), (size[0]*2,size[1]), "parentResponses")
		print "after saving all parent emotional feedback"
		'''

		# 2. train the child on the outputs from the parent	
		nrVisible = len(parentResponses[0])
		nrHidden = 800
		activationFunction = Sigmoid()

		print "visible1 ",nrVisible
		print "data1 ",parentResponses.shape
		print "data row1 ",parentResponses[0,:].shape 

		childNet = rbm.RBM(nrVisible, nrHidden, 0.01, 0.8, 0.8,
						visibleActivationFunction=activationFunction,
						hiddenActivationFunction=activationFunction,
						rmsprop=True,#args.rbmrmsprop,
						nesterov=True,#args.rbmnesterov,
						sparsityConstraint=False,#args.sparsity,
						sparsityRegularization=0.5,
						trainingEpochs=15,#args.maxEpochs,
						sparsityTraget=0.01,
						fixedLabel = True)

		childNet.train(parentResponses)

	if argsdict["trainChild"] != False:
		with open(argsdict["trainChild"], "wb") as f:
				pickle.dump(childNet, f)
				#pickle.dump(parentResponses, f)


	# generate emotions from network trained on parent emotional feedback
	for key in childDataSetOfEmotions.iterkeys():
		print "child Emotion ",key
		emotion = childDataSetOfEmotions[key]
		emotion = emotion.reshape(1, emotion.shape[0]) 
		recon = np.concatenate((rando, emotion), axis=1)
		print "recon shape ",recon.shape
		recon = childNet.reconstruct(recon,30)
		outputEmotions[key] = recon[:,:size[0]**2]# first half is reconstructed bit
		saveImage(recon, key+"afterParent1",(size[0]*2,size[1]), "parentchildoutput")

	# 3. classify the resulting emotion
	return outputEmotions

def scikitclassifier():
	X, Y = readKanade.readAllEmotionssk()

	'''
	Xtest = X[-1]
	Ytest = Y[-1]
	X = X[:-1]
	Y = Y[:-1]
	print("X",X.shape)
	print("Y",Y.shape)
	X = (X - np.min(X, 0)) / (np.max(X, 0) + 0.0001)  # 0-1 scaling

	X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
	                                                    test_size=0.2,
	                                                    random_state=0)
	'''
	# Training Logistic regression
	logistic_classifier = linear_model.LogisticRegression(C=100.0)
	logistic_classifier.fit(X, Y)

	###############################################################################
	# Evaluation

	'''
	print("Logistic regression using raw pixel features:\n%s\n" % (
	    metrics.classification_report(
	        Y_test,
	        logistic_classifier.predict(X_test))))
	'''
	return logistic_classifier

def trainEmotionClassifier():
	# data contains labels as well 
	data, labels = readKanade.readAllEmotions()

	nrVisible = len(data[0])
	nrHidden = 80
	activationFunction = Sigmoid()

	print "visible ",nrVisible
	print "data ",data.shape
	print "data row ",data[0,:] 

	net = rbm.RBM(nrVisible, nrHidden, 0.01, 0.8, 0.8,
					visibleActivationFunction=activationFunction,
					hiddenActivationFunction=activationFunction,
					rmsprop=True,#args.rbmrmsprop,
					nesterov=True,#args.rbmnesterov,
					sparsityConstraint=False,#args.sparsity,
					sparsityRegularization=0.5,
					trainingEpochs=45,#args.maxEpochs,
					sparsityTraget=0.01,
					fixedLabel = True)

	net.train(data[:80,:])
	return net, labels

def runEmoEval(Classifier, emoLabels, emotions):
	for emote in emotions.iterkeys():
		emotion = emotions[emote][:,:size[0]**2]# first half is reconstructed bit

		rando = np.random.random_sample(10)
		rando = rando.reshape(1, 10)
		print "emote ",emotion.shape
		print "rando ",rando.shape
		recon = np.concatenate((emotion, rando), axis=1)
		recon = Classifier.reconstruct(recon, 3000)

		label = recon[:,size[0]**2:]

		print "returned label ", label
		'''
		if label in emoLabels.keys():
			print "expected emotion: ",emote
			print "reconstruct returned: ",emoLabels[label].tolist()
		else:
			for key, emotion in list.iteritems():
					if emotion == emote:
						print "expected label: ",key	
			print "reconstruct returned: ",label 
		'''

def emoEval(classifier, emotions):
	emotionsdct = {1:"fear", 2:"happy",3:"anger",4:"contempt",5:"disgust",6:"sadness",7:"surprise"}
	for emote in emotions.iterkeys():	
		emotion = emotions[emote]
		saveImage(emotion, emote+"FINAL",(50, 50), "parentchildoutput")

		#print "Emotion: ",emotionsdct[classifier.predict(emotion)]
		prediction = classifier.predict(emotion)
		print "Emotion: ",prediction
		print "Expected ",emote," ",emotionsdct[prediction[0]]

def saveImage(data, name, size,temp=""):
	plt.imshow(vectorToImage(data, size), cmap=plt.cm.gray)
	plt.axis('off')
	if temp == "":
		plt.savefig("dump/"+name + '.png',transparent=True) 
	else:
		if(not os.path.exists("dump/"+temp+"/")):
			os.makedirs("dump/"+temp+"/")
		plt.savefig("dump/"+ temp + "/"+name + '.png',transparent=True) 

def main():

	sadnessSecure = {"happy": 0.8, "surprise":0.2}
	happySecure = {"sadness":1}
	childEmotions = {"happy": happySecure, "sadness": sadnessSecure}
	childLabels = {"happy": None, "sadness": None}
	childEmotionalProportions = {"happy": 0.4, "sadness": 0.6}

	net,childLabels = buildParent(childEmotions)

	'''
	for key in childLabels.iterkeys():
		emotion = childLabels[key]
		saveImage(emotion, key+"ceprimim",(25,25), "parentchildoutput")
	'''
	emotionResponses = interactChild(net, childLabels)
	#emoClassifier, emoLabels = trainEmotionClassifier()

	emoEval(scikitclassifier(), emotionResponses)
	#runEmoEval(emoClassifier, emoLabels, emotionResponses)
	print "exited"

if __name__ == '__main__':
	main()