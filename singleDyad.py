import restrictedBoltzmannMachine as rbm
import numpy as np
import cPickle as pickle
import readKanade
import matplotlib.pyplot as plt
from common import *
import os, argparse
from activationfunctions import *

parser = argparse.ArgumentParser(description='dyad simulation')
parser.add_argument('--saveParent',dest='saveParent',action='store_true', default=False,
					help="if true, the data returnet from parent network is saved")
parser.add_argument('--loadParent',dest='loadParent',action='store_true', default=False,
					help="if true, the data from parent network is loaded")
parser.add_argument('--saveNet',dest='saveNet',action='store_true', default=False,
					help="if true, the parent network is saved")
parser.add_argument('--loadNet',dest='loadNet',action='store_true', default=False,
					help="if true, the parent network is loaded")
parser.add_argument('--tempData', dest="tempData", help="file where the parent data should be saved")
parser.add_argument('--tempNet',dest="tempNet", help="file where the network should be saved")

args = parser.parse_args()

small_size = (25,25)
large_size = (50,50)
size = large_size

def buildParent(inputEmotions): # trains network with emotional associations	
	if args.loadNet:
		print "in load"
		f = open(args.tempNet, "rb")
		net = pickle.load(f)
		childLabelList = pickle.load(f)
		f.close()
		print "out of load net"
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


		net = rbm.RBM(nrVisible, nrHidden, 0.01, 0.8, 0.8,
						visibleActivationFunction=activationFunction,
						hiddenActivationFunction=activationFunction,
						rmsprop=True,#args.rbmrmsprop,
						nesterov=True,#args.rbmnesterov,
						sparsityConstraint=False,#args.sparsity,
						sparsityRegularization=0.5,
						trainingEpochs=15,#args.maxEpochs,
						sparsityTraget=0.01,
						fixedLabel = True)

		net.train(finalTrainingData)
		t = visualizeWeights(net.weights.T, (size[0]*2,size[1]), (10,10))
		plt.imshow(t, cmap=plt.cm.gray)
		plt.axis('off')
		plt.savefig('dump/weights.png', transparent=True)

	if args.saveNet:
		with open(args.tempNet, "wb") as f:
				pickle.dump(net, f)
				pickle.dump(childLabelList, f)
	return net, childLabelList

def saveImageFoo(data, name, size):
  plt.imshow(vectorToImage(data, size), cmap=plt.cm.gray)
  plt.axis('off')
  plt.savefig("dump/"+name + '.png',transparent=True) 

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
	# generate sample images as returned from parent net for each child emotion
	for key in childDataSetOfEmotions.iterkeys():
		emotion = childDataSetOfEmotions[key]
		emotion = emotion.reshape(1, emotion.shape[0])
		recon = np.concatenate((rando, emotion), axis=1)
		recon = parentNet.reconstruct(recon,300)
		saveImage(recon, key,(size[0]*2,size[1]), "parentchildoutput")	
	'''
	print "enter interact"

	if args.loadParent: # load previously generated parent emotional feedback database
		f = open(args.tempData, "rb")
		parentResponses = pickle.load(f)
		f.close()
	else:# generate new parent emotional feedback database
		for key in childDataSetOfEmotions.iterkeys():
			for s in xrange(sizeOfParentFeedback):
				emotion = childDataSetOfEmotions[key]
				emotion = emotion.reshape(1, emotion.shape[0])
				recon = np.concatenate((rando, emotion), axis=1)
				recon = parentNet.reconstruct(recon,30)
				if parentResponses.size == 0:
					parentResponses = recon
				else:
					parentResponses = np.vstack((parentResponses, recon))
				#saveImage(recon, key,(size[0]*2,size[1]), "parentchildoutput")

	if args.saveParent:
		with open(args.tempData, "wb") as f:
				pickle.dump(parentResponses, f)

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
	print "data1 ",data.shape
	print "data row1 ",data[0,:].shape 

	net = rbm.RBM(nrVisible, nrHidden, 0.01, 0.8, 0.8,
					visibleActivationFunction=activationFunction,
					hiddenActivationFunction=activationFunction,
					rmsprop=True,#args.rbmrmsprop,
					nesterov=True,#args.rbmnesterov,
					sparsityConstraint=False,#args.sparsity,
					sparsityRegularization=0.5,
					trainingEpochs=15,#args.maxEpochs,
					sparsityTraget=0.01,
					fixedLabel = True)

	net.train(parentResponses)

	# generate emotions from network trained on parent emotional feedback
	for key in childDataSetOfEmotions.iterkeys():
		print "child Emotion ",key
		emotion = childDataSetOfEmotions[key]
		emotion = emotion.reshape(1, emotion.shape[0]) 
		recon = np.concatenate((rando, emotion), axis=1)
		print "recon shape ",recon.shape
		recon = net.reconstruct(recon,300)
		outputEmotions[key] = recon
		saveImage(recon, key+"afterParent1",(size[0]*2,size[1]), "parentchildoutput")

	# 3. classify the resulting emotion
	return outputEmotions

def trainEmotionClassifier():
	# data contains labels as well 
	data, labels = readKanade.readAllEmotions()

	nrVisible = len(data)
	nrHidden = 80
	activationFunction = Sigmoid()

	print "visible ",nrVisible
	print "data ",data.shape
	print "data row ",data[0,:].shape 

	net = rbm.RBM(nrVisible, nrHidden, 0.01, 0.8, 0.8,
					visibleActivationFunction=activationFunction,
					hiddenActivationFunction=activationFunction,
					rmsprop=True,#args.rbmrmsprop,
					nesterov=True,#args.rbmnesterov,
					sparsityConstraint=False,#args.sparsity,
					sparsityRegularization=0.5,
					trainingEpochs=35,#args.maxEpochs,
					sparsityTraget=0.01,
					fixedLabel = True)

	net.train(data)
	return net, labels

def runEmoEval(Classifier, emoLabels, emotions):
	for emote in emotions.iterkeys():
		rando = np.random.random_sample(7)
		recon = np.concatenate((rando, emotion), axis=1)
		recon = Classifier.reconstruct(recon, 30)

		label = recon[:,50**2:]

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

	sadnessSecure = {"happy": 1}
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
	emoClassifier, emoLabels = trainEmotionClassifier()


	runEmoEval(emoClassifier, emoLabels, emotionResponses)
	print "exited"

if __name__ == '__main__':
	main()