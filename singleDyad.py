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
def buildParent(inputEmotions): # trains network with emotional associations

	#asoc = {"fear": "happy", "anger":"fear", "happy":"anger"}
	#childLabelList = {"happy":None, "anger":None, "fear":None}
	data,labels,childLabelList = readKanade.readProportion(inputEmotions)
	print "data.shape"
	print data.shape
	print "labels.shape",labels.shape
	#data = data / 255.0
	#labels = labels / 255.0

	activationFunction = Sigmoid()

	Data = np.concatenate((data, labels), axis=1)
	np.random.shuffle(Data)
	finalTrainingData = Data[0:-1, :]

	nrVisible = len(finalTrainingData[0])
	nrHidden = 800

	net = rbm.RBM(nrVisible, nrHidden, 0.01, 1, 1,
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
	t = visualizeWeights(net.weights.T, (50,25), (10,10))
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
	rando = np.random.random_sample((25,25))
	rando = rando.reshape(1, 625)
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
		saveImage(recon, key,(50,25), "parentchildoutput")

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
	interactChild(net, childLabels)
	print "exited"

if __name__ == '__main__':
	main()