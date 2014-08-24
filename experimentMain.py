import singleDyad as sd
import DoubleDyad as dd
import cPickle as pickle
import os
import itertools
#import numpy as np 
#import csv

parentEmotions = {0:"happy", 1:"anger", 2:"sadness"}#,3:"fear"}
allEmotions = {"fear":0, "happy":1,"anger":2,"contempt":3,"disgust":4,"sadness":5,"surprise":6}

def runExp1(parentTypes, childEmotionalProportions,doComplexFlag):
	nrTrials = 0.0
	maxTrials = 100.0
	justStarting = True

	childEmotionsConv = {}
	emotionRecogRates = {}

	for pType in parentTypes.iterkeys():
		# for each type of parent 

		parentType = parentTypes[pType]
		print parentType
		for key in parentType.iterkeys():
			foo = {}
			#print key
			for pEmo in parentEmotions.iterkeys():
				parentEmotion = parentEmotions[pEmo]
				foo[parentEmotion] = parentType[key][pEmo]
				#print "parent Emo: ",parentEmotion
				#print "prop: ",foo[parentEmotion]
			childEmotionsConv[key] = foo
			parentType[key] = [0,0,0,0,0,0,0] # assign an empty counter for each child emotion in parent type

		#setup dyad parameters and run the singledyad a number of times
		# to return percentages of classifications
		
		if os.path.isfile(pType) and justStarting:
			print "resuming..................."
			f = open(pType, "rb")
			nrTrials = pickle.load(f)
			parentType = pickle.load(f)
			f.close()

		while True:
			print "parent type ",pType
			print "Trial",nrTrials
			trial = sd.run(childEmotionsConv, childEmotionalProportions, doComplexFlag)
			print trial
			for childEmotion in trial.iterkeys():
				trialRecognitionVal = trial[childEmotion]
				positionInRecogList = allEmotions[trialRecognitionVal]
				#print parentType
				#print type(parentType[childEmotion]) 
				parentType[childEmotion][positionInRecogList] += 1
			nrTrials += 1.0
			if nrTrials >= maxTrials:
				break
			with open(pType, "wb") as f:
				pickle.dump(nrTrials, f)
				pickle.dump(parentType, f)
				justStarting = False
				f.close()

	 
		parentType.update((x, [a / nrTrials for a in y]) for x, y in parentType.items())

		emotionRecogRates[pType] = parentType

		# INTRODUCE RESUMABLABLE EXPERIMENT TRIAL

	return emotionRecogRates


def runExp2(parent1, parent2, childEmotionalProportions, parentPercentages):
	childEmotionsConv1 = {}
	childEmotionsConv2= {}
	childEmotions = []
	childEmotionProp = {}
	parentReactionProp = {}

	for key in parent1.iterkeys():
			foo = {}
			#print key
			for pEmo in parentEmotions.iterkeys():
				parentEmotion = parentEmotions[pEmo]
				foo[parentEmotion] = parent1[key][pEmo]
				#print "parent Emo: ",parentEmotion
				#print "prop: ",foo[parentEmotion]
			childEmotionsConv1[key] = foo
			parent1[key] = [0,0,0,0,0,0,0] # assign an empty counter for each child emotion in parent type

	for key in parent2.iterkeys():
			foo = {}
			#print key
			for pEmo in parentEmotions.iterkeys():
				parentEmotion = parentEmotions[pEmo]
				foo[parentEmotion] = parent2[key][pEmo]
				#print "parent Emo: ",parentEmotion
				#print "prop: ",foo[parentEmotion]
			childEmotionsConv2[key] = foo
			parent2[key] = [0,0,0,0,0,0,0] # assign an empty counter for each child emotion in parent type


	dd.run(childEmotionsConv1, childEmotionsConv2, {}, parentPercentages)

def makeParentSpecs():
	def frange(x, y, jump):
	  while round(x,1) <= y:
	    yield round(x,1)
	    x += jump

	securestart = [(0.8,0.9), (0.0,0.1), (0.0,0.1)]
	avoidantstart = [(0.1,0.2), (0.4,0.5), (0.4,0.5)]
	ambivalentstart = [(0.3,0.4), (0.3,0.4), (0.3,0.4)]
	startlist = [securestart, avoidantstart, ambivalentstart]

	outlist = [[], [], []]
	finaloutlist = [[], [], []]

	for start in xrange(len(startlist)):
		foo = []
		#print start
		for j in frange(startlist[start][0][0],startlist[start][0][1], 0.1):
			for k in frange(startlist[start][1][0],startlist[start][1][1], 0.1):
				for l in frange(startlist[start][2][0],startlist[start][2][1], 0.1):
					foo.append(j)
					foo.append(k)
					foo.append(l)
					if sum(foo) == 1.0:
						outlist[start].append(foo)
						print foo
					foo = []

	# convert permutation lists to lists of dictionaries
	for j in xrange(len(outlist)):
		print j
		#finaloutlist.append([])
		for i in itertools.combinations(outlist[j],2):
			dictionar = {"happy":i[0], "sadness":i[1]}
			print dictionar
			finaloutlist[j].append(dictionar)

	return finaloutlist[0], finaloutlist[1], finaloutlist[2]

def main():
	justStarting = True
	# parent emotions
	#sadnessSecure = {"happy": 0.2, "surprise":0.8}
	#happySecure = {"sadness":1}
	
	#secureParent = {"happy":[0.1,0.0,0.9], "sadness":[0.9,0.1, 0.0]}#,"anger":[0.0,0.9,0.0,0.1,0.0]}
	#ambivalentParent = {"happy":[0.4,0.4,0.1], "sadness":[0.3, 0.4, 0.2]}#, "anger":[]}
	#avoidantParent = {"happy":[0.1, 0.4, 0.4], "sadness":[0.1, 0.5, 0.3]}#, "anger":[]}

	secure, avoid, ambiv = makeParentSpecs()
	print "AMBIV", ambiv

	#childEmotions = {"happy": happySecure, "sadness": sadnessSecure}
	childEmotionalProportions = {"happy": 0.4, "sadness": 0.6}
	
	'''
	with open('test_file.csv', 'w') as csvfile:
		writer = csv.writer(csvfile)
		[writer.writerow(r) for r in secureParent.itervalues()]
	'''
	'''
	with open("experiment10Trial.txt", "a") as f:

		f.write("SECURE PARENT\n")
		f.write("######################################################\n")
		resume = -1
		if os.path.isfile("secure") and justStarting:
			fich = open("secure", "rb")
			resume = pickle.load(fich)
			print "resuming...................", resume
			#parentType = pickle.load(fich)
			fich.close()

		for indx in xrange(resume+1, len(secure)):
			config = secure[indx]
			print "Current Config: ", config
			f.write("Parent Responses: \n")
			for k,v in config.iteritems():
				f.write("To "+ k +"\t")
				for i in xrange(len(v)):
					f.write(parentEmotions[i]+": "+ str(v[i]) + " ")
				f.write("\n")
			f.write("\n")
			
			#f.write(str(config)+"\n")

			exp1 = runExp1({"secureParent":config}, childEmotionalProportions,False)
			#justStarting = False
			f.write("Interaction results: \n")
			for k,v in exp1.iteritems():
				#f.write("To "+ k +"\t")
				for j,k in v.iteritems():
					f.write(j+": ")
					f.write("\n")
					for x in xrange(len(k)):
						for key, val in allEmotions.iteritems():
							if val == x:
								f.write(key+": "+str(k[x])+"  ")
					f.write("\n")
			f.write("\n")
			#f.write(str(exp1)+"\n\n")
			f.write("\n\n")

			with open("secure", "wb") as fich:
				pickle.dump(indx, fich)
				#pickle.dump(parentType, fich)
				justStarting = False
				fich.close()

	with open("experiment10Trial.txt", "a") as f:
		f.write("AMBIVALENT PARENT\n")
		f.write("######################################################\n")
		resume = -1
		justStarting = True
		if os.path.isfile("ambivalent") and justStarting:
			fich = open("ambivalent", "rb")
			resume = pickle.load(fich)
			print "resuming...................", resume
			#parentType = pickle.load(fich)
			fich.close()

		for indx in xrange(resume+1, len(ambiv)):
			config = ambiv[indx]
			print "Current Config: ", config
			f.write("Parent Responses: \n")
			for k,v in config.iteritems():
				f.write("To "+ k +"\t")
				for i in xrange(len(v)):
					#print v
					#print len(v)
					#print i
					f.write(parentEmotions[i]+": "+ str(v[i]) + " ")
				f.write("\n")
			f.write("\n")
			
			#f.write(str(config)+"\n")

			exp2 = runExp1({"ambivalentParent":config}, childEmotionalProportions,False)
			f.write("Interaction results: \n")
			for k,v in exp2.iteritems():
				#f.write("To "+ k +"\t")
				for j,k in v.iteritems():
					f.write(j+": ")
					f.write("\n")
					for x in xrange(len(k)):
						for key, val in allEmotions.iteritems():
							if val == x:
								f.write(key+": "+str(k[x])+"  ")
					f.write("\n")
			f.write("\n")
			#f.write(str(exp2)+"\n\n")
			f.write("\n\n")
			with open("ambivalent", "wb") as fich:
				pickle.dump(indx, fich)
				#pickle.dump(parentType, fich)
				justStarting = False
				fich.close()

	with open("experiment10Trial.txt", "a") as f:
		f.write("AVOIDANT PARENT\n")
		f.write("######################################################\n")

		resume = -1
		justStarting = True
		if os.path.isfile("avoidant") and justStarting:
			fich = open("avoidant", "rb")
			resume = pickle.load(fich)
			print "resuming...................", resume
			#parentType = pickle.load(fich)
			fich.close()

		for indx in xrange(resume+1, len(avoid)):
			config = avoid[indx]
			print "Current Config: ", config
			f.write("Parent Responses: \n")
			for k,v in config.iteritems():
				f.write("To "+ k +"\t")
				for i in xrange(len(v)):
					f.write(parentEmotions[i]+": "+ str(v[i]) + " ")
				f.write("\n")
			f.write("\n")
			
			#f.write(str(secureParent)+"\n")

			exp3 = runExp1({"avoidantParent":config}, childEmotionalProportions,False)

			f.write("Interaction results: \n")
			for k,v in exp3.iteritems():
				#f.write("To "+ k +"\t")
				for j,k in v.iteritems():
					f.write(j+": ")
					f.write("\n")
					for x in xrange(len(k)):
						for key, val in allEmotions.iteritems():
							if val == x:
								f.write(key+": "+str(k[x])+"  ")
					f.write("\n")
			f.write("\n")
			#f.write(str(exp3)+"\n\n")
			f.write("\n\n")
			with open("avoidant", "wb") as fich:
				pickle.dump(indx, fich)
				#pickle.dump(parentType, fich)
				justStarting = False
				fich.close()
	'''
	#print runExp1({"secureParent":secureParent}, childEmotionalProportions)
	#print runExp1({"ambivalentParent":ambivalentParent}, childEmotionalProportions)
	#print runExp1({"avoidantParent":avoidantParent}, childEmotionalProportions)

	print runExp2(secureParent, ambivalentParent, {}, [0.6, 0.4])

if __name__ == '__main__':
	main()