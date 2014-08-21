import singleDyad as sd
import numpy as np 

def runExp1(parentTypes, childEmotionalProportions):
	parentEmotions = {0:"happy", 1:"anger", 2:"sadness",3:"fear"}#,4:"surprise"}
	allEmotions = {"fear":0, "happy":1,"anger":2,"contempt":3,"disgust":4,"sadness":5,"surprise":6}
	childEmotionsConv = {}
	emotionRecogRates = {}

	for pType in parentTypes.iterkeys():
		# for each type of parent 
		nrTrials = 3.0
		rates = {"fear":0, "happy":0,"anger":0,"contempt":0,"disgust":0,"sadness":0,"surprise":0}
		parentType = parentTypes[pType]
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
		
		for i in xrange(int(nrTrials)):
			trial = sd.run(childEmotionsConv, childEmotionalProportions)
			print trial
			for childEmotion in trial.iterkeys():
				trialRecognitionVal = trial[childEmotion]
				positionInRecogList = allEmotions[trialRecognitionVal]
				#print parentType
				#print type(parentType[childEmotion]) 
				parentType[childEmotion][positionInRecogList] += 1

		parentType.update((x, [a / nrTrials for a in y]) for x, y in parentType.items())

		emotionRecogRates[pType] = parentType

	return emotionRecogRates

def runExp2():
	childEmotions = []
	childEmotionProp = {}
	parentReactionProp = {}
	sd.main()

def main():

	# parent emotions
	#sadnessSecure = {"happy": 0.2, "surprise":0.8}
	#happySecure = {"sadness":1}
	secureParent = {"happy":[0.1,0.0,0.9,0.0], "sadness":[0.9,0.1, 0.0, 0.0]}#,"anger":[0.0,0.9,0.0,0.1,0.0]}
	ambivalentParent = {"happy":[0.4,0.4,0.1,0.1], "sadness":[0.3, 0.4, 0.2,0.1]}#, "anger":[]}
	avoidantParent = {"happy":[0.1, 0.4, 0.4,0.1], "sadness":[0.1, 0.5,0.3,0.1]}#, "anger":[]}

	#childEmotions = {"happy": happySecure, "sadness": sadnessSecure}
	childEmotionalProportions = {"happy": 0.4, "sadness": 0.6}


	with open("experiment1.txt", "w") as f:
		f.write("SECURE PARENT\n")
		f.write("######################################################\n")
		f.write(str(secureParent)+"\n")
		exp1 = runExp1({"secureParent":secureParent}, childEmotionalProportions)
		f.write(str(exp1)+"\n\n")

		f.write("AMBIVALENT PARENT\n")
		f.write("######################################################\n")
		f.write(str(ambivalentParent)+"\n")
		f.write(str(runExp1({"ambivalentParent":ambivalentParent}, childEmotionalProportions))+"\n\n")

		f.write("AVOIDANT PARENT\n")
		f.write("######################################################\n")
		f.write(str(avoidantParent)+"\n")
		f.write(str(runExp1({"avoidantParent":avoidantParent}, childEmotionalProportions))+"\n\n")

	#print runExp1({"secureParent":secureParent}, childEmotionalProportions)

	#print runExp1({"ambivalentParent":ambivalentParent}, childEmotionalProportions)

	#print runExp1({"avoidantParent":avoidantParent}, childEmotionalProportions)

if __name__ == '__main__':
	main()