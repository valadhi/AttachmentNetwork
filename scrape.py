''''
def trainEmotionClassifier():
	# data contains labels as well 
	data, labels = readKanade.readAllEmotions()

	nrVisible = len(data[0])
	nrHidden = 800
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
'''
'''
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

		
		if label in emoLabels.keys():
			print "expected emotion: ",emote
			print "reconstruct returned: ",emoLabels[label].tolist()
		else:
			for key, emotion in list.iteritems():
					if emotion == emote:
						print "expected label: ",key	
			print "reconstruct returned: ",label 
		
'''