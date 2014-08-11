import restrictedBoltzmannMachine as rbm
import numpy as np
import readKanade
import matplotlib.pyplot as plt
from common import *
import os
from activationfunctions import *

'''parser = argparse.ArgumentParser(description='dyad simulation')
parser.add_argument('--save',dest='save',action='store_true', default=False,
                    help="if true, the network is serialized and saved")
'''

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

	print type(data)
	print type(labels)
	data = data / 255.0
	labels = labels / 255.0
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
					trainingEpochs=45,#args.maxEpochs,
					sparsityTraget=0.01,
					fixedLabel = True)

	net.train(finalTrainingData)
	t = visualizeWeights(net.weights.T, (size[0]*2,size[1]), (10,10))
	plt.imshow(t, cmap=plt.cm.gray)
	plt.axis('off')
	plt.savefig('dump/weights.png', transparent=True)

	return net, childLabelList

def saveImageFoo(data, name, size):
  plt.imshow(vectorToImage(data, size), cmap=plt.cm.gray)
  plt.axis('off')
  plt.savefig("dump/"+name + '.png',transparent=True) 

def interactChild(parentNet, childDataSetOfEmotions):
	# 1. feed the child emotions into parent
	rando = np.random.random_sample(size)
	rando = rando.reshape(1, size[0]**2)
	#aveImage(rando, "rando_",(25,25), "parentchildoutput")	
	#print rando

	for key in childDataSetOfEmotions.iterkeys():
		emotion = childDataSetOfEmotions[key]
		#print type(emotion)
		#print "emotion shape",emotion.shape
		#print "rando shape",rando.shape
		emotion = emotion.reshape(1, emotion.shape[0])

		recon = np.concatenate((rando, emotion), axis=1)
		print "RECON",recon
		#saveImage(recon, key+"_",(50,25), "parentchildoutput")
		#print "RANDOM ########## ",emotion		
		recon = parentNet.reconstruct(recon,300)
		saveImage(recon, key,(size[0]*2,size[1]), "parentchildoutput")

	# 2. train the child on the outputs from the parent	
	# 3. classify the resulting emotion
	return None

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
	happySecure = {"happy":1, "anger":0.1, "surprise":0.1}
	childEmotions = {"happy": happySecure, "sadness": sadnessSecure}
	childLabels = {"happy": None, "sadness": None}
	childEmotionalProportions = {"happy": 0.4, "sadness": 0.6}

	net,childLabels = buildParent(childEmotions)


	'''
	for key in childLabels.iterkeys():
		emotion = childLabels[key]
		saveImage(emotion, key+"ceprimim",(25,25), "parentchildoutput")
	'''
	interactChild(net, childLabels)
	print "exited"

if __name__ == '__main__':
	main()