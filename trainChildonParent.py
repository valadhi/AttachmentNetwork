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
import os.path
import numpy as np
import cv2
import random,copy
import matplotlib.pyplot as plt
from common import *


large_size = (50,50)
size = large_size
def parseImage(path):
	img = cv2.imread(path, 0)
	img = cv2.resize(img, (100,50))
	img = cv2.equalizeHist(img)
	img = img.flatten()
	return img.tolist()
	'''img = Image.open(path)
	pixels = img.load()
	array = []
	for i in range(img.size[0]):
		for j in range(img.size[1]):
			(a,b,c,d) = pixels[i,j]
			array.append((a+b+c)/3)    
	return array'''
def readEmotion(emo):
	out = []
	for pic in os.listdir("dump/"+emo):
		img = parseImage("dump/"+emo+"/"+pic)
		print pic
		out.append(img)
		print len(img)
	return np.array(out)

def buildParent(): # trains network with emotional associations	
	'''
	if argsdict["loadParent"] != False:
		print "in load parentNet"
		f = open(argsdict["loadParent"], "rb")
		savedinput = pickle.load(f)
		parentNet = savedinput["net"]
		childLabelList = savedinput["labels"]
		f.close()
		print "out of load parentNet"
	else:
	'''
	#asoc = {"fear": "happy", "anger":"fear", "happy":"anger"}
	#childLabelList = {"happy":None, "anger":None, "fear":None}
	finalTrainingData = readEmotion("parentResponses")
	print finalTrainingData.shape
	
	saveImage(finalTrainingData[0], "afterParent2",(size[0],size[1]*2), "parentchildoutput")
	saveImage(finalTrainingData[0][:size[0]**2], "afterParent1",(size[0],size[1]), "parentchildoutput")
	
	activationFunction = Sigmoid()
	nrVisible = 5000
	print nrVisible
	nrHidden = 800


	parentNet = rbm.RBM(nrVisible, nrHidden, 0.01, 0.8, 0.8,
					visibleActivationFunction=activationFunction,
					hiddenActivationFunction=activationFunction,
					rmsprop=True,#args.rbmrmsprop,
					nesterov=True,#args.rbmnesterov,
					sparsityConstraint=False,#args.sparsity,
					sparsityRegularization=0.5,
					trainingEpochs=1,#args.maxEpochs,
					sparsityTraget=0.01,
					fixedLabel = True)

	parentNet.train(finalTrainingData)
	t = visualizeWeights(parentNet.weights.T, (size[0]*2,size[1]), (10,10))
	plt.imshow(t, cmap=plt.cm.gray)
	plt.axis('off')
	plt.savefig('dump/weights.png', transparent=True)

	childDataSetOfEmotions = {"hap":finalTrainingData[0][size[0]**2:], "sad":finalTrainingData[-1][size[0]**2:]}
	rando = np.random.random_sample(size)
	rando = rando.reshape(1, size[0]**2)
	for key in childDataSetOfEmotions.iterkeys():
		print "child Emotion ",key
		emotion = childDataSetOfEmotions[key]
		emotion = emotion.reshape(1, emotion.shape[0]) 
		recon = np.concatenate((rando, emotion), axis=1)
		print "recon shape ",recon.shape
		recon = parentNet.reconstruct(recon,30)
		#outputEmotions[key] = recon[:,:size[0]**2]# first half is reconstructed bit
		saveImage(recon, key+"afterParent1",(size[0],size[1]*2), "parentchildoutput")
	'''
	if argsdict["trainParent"] != False:
		print "saving Parent Net"
		with open(argsdict["trainParent"], "wb") as f:
			pickle.dump({"net":parentNet, "labels":childLabelList}, f)
			#pickle.dump(childLabelList, f)
	'''
def saveImage(data, name, size,temp=""):
	print "data ",data.shape
	print "size ",size
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

buildParent()