import singleDyad as sd
import DoubleDyad as dd
import cPickle as pickle
import os
import itertools
import time
import copy
#import numpy as np 
#import csv

parentEmotions = {0:"happy", 1:"anger", 2:"sadness"}#,3:"fear"}
allEmotions = {"fear":0, "happy":1,"anger":2,"contempt":3,"disgust":4,"sadness":5,"surprise":6}


##### parentType = {"sadness":[0.1,0.2,0.7], "happy":[0.3,0.4,0.3]}
def runExp1(parentTypes, childEmotionalProportions,doComplexFlag):

	nrTrials = 0.0
	maxTrials = 80.0
	#justStarting = True

	childEmotionsConv = {}
	emotionRecogRates = {}

	for pType in parentTypes.iterkeys():
		# for each type of parent 

		parentType = parentTypes[pType]
		outparent = {}

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
			outparent[key] = [0,0,0,0,0,0,0] # assign an empty counter for each child emotion in parent type

		#setup dyad parameters and run the singledyad a number of times
		# to return percentages of classifications
		
		if os.path.isfile(pType):
			print "resuming..................."
			f = open(pType, "rb")
			nrTrials = pickle.load(f)
			outparent = pickle.load(f)
			print nrTrials
			print outparent
			f.close()

		while nrTrials < maxTrials:
			print "parent type ",pType
			print "Trial",nrTrials
			trial = sd.run(childEmotionsConv, childEmotionalProportions, doComplexFlag)
			#trial = {"sadness":"anger", "happy":"sadness"}
			#time.sleep(0.1)
			print trial
			for childEmotion in trial.iterkeys():
				#print childEmotion
				trialRecognitionVal = trial[childEmotion]
				positionInRecogList = allEmotions[trialRecognitionVal]
				#print outparent
				#print type(outparent[childEmotion]) 
				outparent[childEmotion][positionInRecogList] += 1
			print outparent
			nrTrials += 1.0
			#if nrTrials >= maxTrials:
			#	break
			if nrTrials == maxTrials:# reset trial and outparent objects
				print "ended configuration for "+str(nrTrials)+" trials on " + pType
				endingtrial = 0.0
				endingparent = {}
				for k in outparent.iterkeys():
					print k
					endingparent[k] = [0,0,0,0,0,0,0]
				print "ENDING PARENT",endingparent
				print endingtrial
				
				with open(pType, "wb") as f:
					pickle.dump(endingtrial, f)
					pickle.dump(endingparent, f)
					#justStarting = False
					f.close()
			else:
				with open(pType, "wb") as f:
					pickle.dump(nrTrials, f)
					pickle.dump(outparent, f)
					#justStarting = False
					f.close()

	 	print "OUTPARENT",outparent
	 	print nrTrials
		outparent.update((x, [a / nrTrials for a in y]) for x, y in outparent.items())

		emotionRecogRates[pType] = outparent
		#print "OUTPARENT",outparent
		#print "OUTPARENT",emotionRecogRates
		# INTRODUCE RESUMABLABLE EXPERIMENT TRIAL

	return emotionRecogRates, nrTrials, parentType


def runExp2(parent1, parent2, childEmotionalProportions, parentPercentages, resumeFile):
	childEmotionsConv1 = {}
	childEmotionsConv2= {}
	childEmotions = []
	childEmotionProp = {}
	parentReactionProp = {}
	nrTrials = 0.0
	maxTrials = 100.0
	#justStarting = True
	outparent = {}

	childEmotionsConv = {}
	emotionRecogRates = {}
	for key in parent1.iterkeys():
			foo = {}
			#print key
			for pEmo in parentEmotions.iterkeys():
				parentEmotion = parentEmotions[pEmo]
				foo[parentEmotion] = parent1[key][pEmo]
				#print "parent Emo: ",parentEmotion
				#print "prop: ",foo[parentEmotion]
			childEmotionsConv1[key] = foo
			outparent[key] = [0,0,0,0,0,0,0] # assign an empty counter for each child emotion in parent type

	for key in parent2.iterkeys():
			foo = {}
			#print key
			for pEmo in parentEmotions.iterkeys():
				parentEmotion = parentEmotions[pEmo]
				foo[parentEmotion] = parent2[key][pEmo]
				#print "parent Emo: ",parentEmotion
				#print "prop: ",foo[parentEmotion]
			childEmotionsConv2[key] = foo
			#parent2[key] = [0,0,0,0,0,0,0] # assign an empty counter for each child emotion in parent type


	#dd.run(childEmotionsConv1, childEmotionsConv2, {}, parentPercentages)
	if os.path.isfile(resumeFile):
			print "resuming..................."
			f = open(resumeFile, "rb")
			nrTrials = pickle.load(f)
			outparent = pickle.load(f)
			print nrTrials
			print outparent
			f.close()

	while nrTrials < maxTrials:
		print "parent type ",resumeFile
		print "Trial",nrTrials
		trial = dd.run(childEmotionsConv1, childEmotionsConv2, {}, parentPercentages)
		#trial = {"sadness":"anger", "happy":"sadness"}
		#time.sleep(0.1)
		print trial
		for childEmotion in trial.iterkeys():
			print childEmotion
			trialRecognitionVal = trial[childEmotion]
			positionInRecogList = allEmotions[trialRecognitionVal]
			print positionInRecogList
			print type(outparent[childEmotion]) 
			outparent[childEmotion][positionInRecogList] += 1
		print outparent
		nrTrials += 1.0
		#if nrTrials >= maxTrials:
		#	break
		if nrTrials == maxTrials:# reset trial and outparent objects
			print "ended configuration for "+str(nrTrials)+" trials on " + resumeFile
			endingtrial = 0.0
			endingparent = {}
			for k in outparent.iterkeys():
				print k
				endingparent[k] = [0,0,0,0,0,0,0]
			print "ENDING PARENT",endingparent
			print endingtrial
			
			with open(resumeFile, "wb") as f:
				pickle.dump(endingtrial, f)
				pickle.dump(endingparent, f)
				#justStarting = False
				f.close()
		else:
			with open(resumeFile, "wb") as f:
				pickle.dump(nrTrials, f)
				pickle.dump(outparent, f)
				#justStarting = False
				f.close()

 	print "OUTPARENT",outparent
 	print nrTrials
	outparent.update((x, [a / nrTrials for a in y]) for x, y in outparent.items())

	emotionRecogRates[resumeFile] = outparent
	#print "OUTPARENT",outparent
	#print "OUTPARENT",emotionRecogRates
	# INTRODUCE RESUMABLABLE EXPERIMENT TRIAL

	return emotionRecogRates, nrTrials, resumeFile

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

def parseExp2(parent1, parent2, childEmotionalProportions, parentPercentages, resumeFileName):
	with open("experimentDoubleDyad.txt", "a") as f:
		f.write(resumeFileName + " DYAD\n")
		f.write("######################################################\n")
		'''
		resume = 0
		if os.path.isfile(resumeFileName):
			fich = open(resumeFileName, "rb")
			resume = pickle.load(fich)
			print "resuming..................."+str(resume)+" out of "+str(len(avoid))
			#parentType = pickle.load(fich)
			fich.close()
		'''

		f.write("Parent 1: "+str(parentPercentages[0])+"\n")
		for k,v in parent1.iteritems():
			f.write("To "+ k +"\t")
			for i in xrange(len(v)):
				f.write(parentEmotions[i]+": "+ str(v[i]) + " ")
			f.write("\n")
		f.write("\n")
		f.write("Parent 2:"+str(parentPercentages[1])+" \n")
		for k,v in parent2.iteritems():
			f.write("To "+ k +"\t")
			for i in xrange(len(v)):
				f.write(parentEmotions[i]+": "+ str(v[i]) + " ")
			f.write("\n")
		f.write("\n")
		
		#f.write(str(secureParent)+"\n")

		#exp3,nrTrials, ptype = runExp1({"avoidantParent":config}, childEmotionalProportions,False)
		exp3,nrTrials, ptype = runExp2(parent1, parent2, childEmotionalProportions, parentPercentages, resumeFileName)
		#justStarting = False
		f.write("Interaction results: \n")
		f.write("Nr Trials: "+ str(nrTrials) + " Percentages: " +str(parentPercentages) +"\n")
		f.write("Parent Type "+ ptype+"\n")
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
		'''
		with open(resumeFileName, "wb") as fich:
			print "###################",str(indx+1)
			pickle.dump(indx+1, fich)
			#pickle.dump(parentType, fich)
			fich.close()
		'''

def main():
	justStarting = True
	# parent emotions
	#sadnessSecure = {"happy": 0.2, "surprise":0.8}
	#happySecure = {"sadness":1}
	
	#secureParent = {"happy":[0.1,0.0,0.9], "sadness":[0.9,0.1, 0.0]}#,"anger":[0.0,0.9,0.0,0.1,0.0]}
	#ambivalentParent = {"happy":[0.4,0.4,0.1], "sadness":[0.3, 0.4, 0.2]}#, "anger":[]}
	#avoidantParent = {"happy":[0.1, 0.4, 0.4], "sadness":[0.1, 0.5, 0.3]}#, "anger":[]}
	'''
	secure, avoid, ambiv = makeParentSpecs()
	print "AMBIV", ambiv
	print "AVOID", avoid
	print "SECURE", secure
	'''
	ambiv = [{"happy":[0.5,0.2,0.3], "sadness":[0.5,0.3,0.2]}, {"happy":[0.4,0.4,0.1], "sadness":[0.4,0.1,0.4]},{"happy":[0.6,0.2,0.2], "sadness":[0.5,0.2,0.3]}]
	avoid = [{"happy":[0.4,0.3,0.3], "sadness":[0.4,0.2,0.4]},{"happy":[0.3,0.4,0.3], "sadness":[0.2,0.3,0.5]},{"happy":[0.3,0.2,0.5], "sadness":[0.3,0.1,0.4]}]
	secure = [{"happy":[0.9,0.0,0.1], "sadness":[0.8,0.1, 0.1]},{"happy":[0.7,0.1,0.2], "sadness":[0.7,0.2,0.1]}]

	#childEmotions = {"happy": happySecure, "sadness": sadnessSecure}
	childEmotionalProportions = {"happy": 0.4, "sadness": 0.6}
	
	'''
	with open('test_file.csv', 'w') as csvfile:
		writer = csv.writer(csvfile)
		[writer.writerow(r) for r in secureParent.itervalues()]
	'''
	
	with open("experiment10Trial.txt", "a") as f:

		f.write("SECURE PARENT\n")
		f.write("######################################################\n")
		resume = 0
		if os.path.isfile("secure"):
			fich = open("secure", "rb")
			resume = pickle.load(fich)
			print "resuming..................."+str(resume)+" out of "+str(len(secure))
			#parentType = pickle.load(fich)
			fich.close()

		for indx in xrange(resume, len(secure)):
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

			exp1,nrTrials, ptype = runExp1({"secureParent":config}, childEmotionalProportions,False)
			#justStarting = False
			f.write("Interaction results: \n")
			f.write("Nr Trials: "+ str(nrTrials) + "\n")
			f.write("Parent Type"+ str(ptype)+"\n")
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
				pickle.dump(indx+1, fich)
				#pickle.dump(parentType, fich)
				#justStarting = False
				fich.close()

	with open("experiment10Trial.txt", "a") as f:
		f.write("AMBIVALENT PARENT\n")
		f.write("######################################################\n")
		resume = 0
		if os.path.isfile("ambivalent"):
			fich = open("ambivalent", "rb")
			resume = pickle.load(fich)
			print "resuming..................."+str(resume)+" out of "+str(len(ambiv))
			#parentType = pickle.load(fich)
			fich.close()

		for indx in xrange(resume, len(ambiv)):
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

			exp2,nrTrials, ptype = runExp1({"ambivalentParent":config}, childEmotionalProportions,False)
			
			#justStarting = False
			f.write("Interaction results: \n")
			f.write("Nr Trials: "+ str(nrTrials) + "\n")
			f.write("Parent Type"+ str(ptype)+"\n")

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
				pickle.dump(indx+1, fich)
				#pickle.dump(parentType, fich)
				fich.close()

	with open("experiment10Trial.txt", "a") as f:
		f.write("AVOIDANT PARENT\n")
		f.write("######################################################\n")

		resume = 0
		if os.path.isfile("avoidant"):
			fich = open("avoidant", "rb")
			resume = pickle.load(fich)
			print "resuming..................."+str(resume)+" out of "+str(len(avoid))
			#parentType = pickle.load(fich)
			fich.close()

		for indx in xrange(resume, len(avoid)):
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

			exp3,nrTrials, ptype = runExp1({"avoidantParent":config}, childEmotionalProportions,False)
			#justStarting = False
			f.write("Interaction results: \n")
			f.write("Nr Trials: "+ str(nrTrials) + "\n")
			f.write("Parent Type"+ str(ptype)+"\n")
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
				print "###################",str(indx+1)
				pickle.dump(indx+1, fich)
				#pickle.dump(parentType, fich)
				fich.close()
	
	#print runExp1({"secureParent":secureParent}, childEmotionalProportions)
	#print runExp1({"ambivalentParent":ambivalentParent}, childEmotionalProportions)
	#print runExp1({"avoidantParent":avoidantParent}, childEmotionalProportions)
	
	'''
	resume = 0
	if os.path.isfile("doubleDyad"):
		fich = open("doubleDyad", "rb")
		resume = pickle.load(fich)
		print "resuming..................."+str(resume)+" out of "+str(len(avoid))
		#parentType = pickle.load(fich)
		fich.close()
	secureParent = {"happy":[0.8,0.1,0.1], "sadness":[0.8,0.1, 0.1]}#,"anger":[0.0,0.9,0.0,0.1,0.0]}
	ambivalentParent = {"happy":[0.5,0.2,0.3], "sadness":[0.5, 0.2, 0.3]}#, "anger":[]}
	avoidantParent = {"happy":[0.3,0.5,0.2], "sadness":[0.3, 0.5, 0.2]}
	testinput = [[secureParent,ambivalentParent,[0.6,0.4],"secureambivalent"]]
	testinput.append([secureParent,avoidantParent,[0.6,0.4],"secureavoidant"])
	testinput.append([ambivalentParent,avoidantParent,[0.6,0.4],"ambivalentavoidant"])
	testinput.append([secureParent,ambivalentParent,[0.4,0.6],"secureambivalent"])
	testinput.append([secureParent,avoidantParent,[0.4,0.6],"secureavoidant"])
	testinput.append([ambivalentParent,avoidantParent,[0.4,0.6],"ambivalentavoidant"])

	for indx in xrange(resume, len(testinput)):
		testdata = testinput[indx]
		parseExp2(testdata[0], testdata[1], {}, testdata[2], resumeFileName = testdata[3])

		with open("doubleDyad", "wb") as fich:
			print "###################",str(indx+1)
			pickle.dump(indx+1, fich)
			#pickle.dump(parentType, fich)
			fich.close()
	'''
	#secureParent = {"happy":[0.9,0.0,0.1], "sadness":[0.9,0.0, 0.1]}#,"anger":[0.0,0.9,0.0,0.1,0.0]}
	#ambivalentParent = {"happy":[0.1,0.9,0.0], "sadness":[0.1, 0.9, 0.0]}#, "anger":[]}
	#print runExp2(secureParent, ambivalentParent, {}, [0.5, 0.5],"secureambivalent")
if __name__ == '__main__':
	main()