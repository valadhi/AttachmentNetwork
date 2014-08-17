import os.path
import numpy as np
import cv2
import random,copy
import matplotlib.pyplot as plt
from common import *

#emotionArray = ["anger","contempt","disgust","fear","happy","sadness","surprise"]
kanadeEmotions = {"anger":1, "contempt":2, "disgust":3, "fear":4, "happy":5, "sadness":6, "surprise":7}

classLabels = {}
totHidden = 3
dataMultiplier = 2
small_size = (25,25)
large_size = (50,50)
size = large_size
sizeName = str(size[0]) + "_" + str(size[1])
pathData = "kanade/"+ str(sizeName) + "/"

#parse images in array bit format
def parseImage(path):
	img = cv2.imread(path, 0)
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
# reads the data for a single emotion
def readEmotion(emo):
	out = []
	for pic in os.listdir("kanade/"+emo):
		out.append(parseImage("kanade/"+emo+"/"+pic))
	return np.array(out)

def genLabel(emotion):
	label = []
	for i in xrange(kanadeEmotions[emotion]):
		label.append(1)
	label = label + [0]*(10 - len(label))
	print "LABEL: ",label
	return label

def readAllEmotions():
	#data = {"fear": [], "happy": [], "anger": [], "contempt": [], "disgust": [], "sadness": [], "surprise": []}
  	data = []
  	labels = {}
  	emotions = ["fear", "happy", "anger", "contempt", "disgust", "sadness", "surprise"]
	for emote in emotions:
		path = pathData + emote + "/"
		labels[emote] = genLabel(emote)
		for pic in os.listdir(path):
			data.append([x / 255.0 for x in parseImage(path+pic)] + 
				[int(x) for x in genLabel(emote)])
			'''
			if(emote == "anger"):
				data["anger"].append(parseImage(path+pic))
			elif(emote == "fear"):
				data["fear"].append(parseImage(path+pic))
			elif(emote == "happy"):
				data["happy"].append(parseImage(path+pic))
			elif(emote == "disgust"):
				data["disgust"].append(parseImage(path+pic))
			elif(emote == "sadness"):
				data["sadness"].append(parseImage(path+pic))
			elif(emote == "surprise"):
				data["surprise"].append(parseImage(path+pic))
			elif(emote == "contempt"):
				data["contempt"].append(parseImage(path+pic))
			'''
	#print "DATA",np.array(data["anger"][0])

	return np.array(data), labels

def readAllEmotionssk():
	#data = {"fear": [], "happy": [], "anger": [], "contempt": [], "disgust": [], "sadness": [], "surprise": []}
  	data = []
  	labels = []
  	emotions = ["fear", "happy","anger","contempt","disgust","sadness","surprise"]
  	emotionsdct = {"fear":1, "happy":2,"anger":3,"contempt":4,"disgust":5,"sadness":6,"surprise":7}
	for emote in emotions:
		path = pathData + emote + "/"
		#labels[emote] = genLabel(emote)
		for pic in os.listdir(path):
			data.append([x / 255.0 for x in parseImage(path+pic)])
			#data.append([0 if x < 127 else 1 for x in parseImage(path+pic)])
			labels.append(emotionsdct[emote])
			'''
			if(emote == "anger"):
				data["anger"].append(parseImage(path+pic))
			elif(emote == "fear"):
				data["fear"].append(parseImage(path+pic))
			elif(emote == "happy"):
				data["happy"].append(parseImage(path+pic))
			elif(emote == "disgust"):
				data["disgust"].append(parseImage(path+pic))
			elif(emote == "sadness"):
				data["sadness"].append(parseImage(path+pic))
			elif(emote == "surprise"):
				data["surprise"].append(parseImage(path+pic))
			elif(emote == "contempt"):
				data["contempt"].append(parseImage(path+pic))
			'''
	#print "DATA",np.array(data["anger"][0])

	return np.array(data), np.array(labels)
# reads the data for the given associations
def readWithAsoc(asociations):
	foldersize = []
	anger = []
	fear = []
	happy = []
	outdata = []
	labels = []
	#add all parsed pictures to the dictionary of emotions
	data = {"anger":anger, "fear": fear, "happy": happy}
	for emote in asociations.iterkeys():
		path = pathData + emote + "/"
		for pic in os.listdir(path):
			if(emote == "anger"):
				anger.append(parseImage(path+pic))
			elif(emote == "fear"):
				fear.append(parseImage(path+pic))
			elif(emote == "happy"):
				happy.append(parseImage(path+pic))							
		foldersize.append(len(os.listdir(path)))

		'''HISTOGRAM '''
	
	for emote in asociations.iterkeys():
		if(emote == "anger"):
			label = anger[0]
		elif(emote == "fear"):
			label = fear[0]
		elif(emote == "happy"):
			label = happy[0]
		for i in data[asociations.get(emote)]:
			outdata.append(i)
			labels.append(label)

	return np.array(outdata), np.array(labels)
	#common = min(foldersize)
	#for i in xrange(common):
def saveImage(data, name, size,temp=""):
	plt.imshow(vectorToImage(data, size), cmap=plt.cm.gray)
	plt.axis('off')
	if temp == "":
		plt.savefig("dump/"+name + '.png',transparent=True) 
	else:
		if(not os.path.exists("dump/"+temp+"/")):
			os.makedirs("dump/"+temp+"/")
		plt.savefig("dump/"+ temp + "/"+name + '.png',transparent=True) 
def readProportion(inputEmotions):
	allData = readAllEmotions()	# could modify to only read the emotions that are needed
	outdata = []
	outlabels = [] 
	finaloutdata = [] 
	finaloutlabels = []
	emotionLabels = {}
	
	for childEmotion in inputEmotions.iterkeys():
		# take first image in emotion folder
		labelPath = pathData + childEmotion + "/"
		labelImage = os.listdir(labelPath)[0]
		currentLabel = parseImage(labelPath + labelImage)
		emotionLabels[childEmotion] = np.array(currentLabel) / 255.0

		sumofemotions = 0
		for parentReaction in inputEmotions[childEmotion].iterkeys():
			sumofemotions += len(os.listdir(pathData + parentReaction))

		maxdiff = 0
		maxemotion = ""
		for parentReaction in inputEmotions[childEmotion].iterkeys():

			desiredParentProportion = inputEmotions[childEmotion][parentReaction]
			availableEmotionData = len(os.listdir(pathData + parentReaction))
			availableEmotionProportion = availableEmotionData/float(sumofemotions)

			if (desiredParentProportion > availableEmotionProportion):
				diff = desiredParentProportion - availableEmotionProportion
				if diff > maxdiff:
					maxdiff = diff
					maxemotion = parentReaction	

		print maxemotion," needs adjusting"	

		for parentReaction in inputEmotions[childEmotion].iterkeys():
			tempdata = [] 
			templabels = []
			imageList = os.listdir(pathData + parentReaction)
			emotionDatasize = len(imageList)
			print "parentReaction", parentReaction
			print "firstImage", imageList[0]
			wantedEmotionSize = inputEmotions[childEmotion][parentReaction]  * emotionDatasize

			if(maxemotion != ""):
				adjustedTotal = len(os.listdir(pathData + maxemotion)) / inputEmotions[childEmotion][maxemotion]
				wantedEmotionSize = int(inputEmotions[childEmotion][parentReaction] * adjustedTotal)
				assert wantedEmotionSize <= emotionDatasize

			assert tempdata == []
			assert templabels == []
			for imageIndx in xrange(wantedEmotionSize): # attach (label, data) for all needed images
				tempdata.append(parseImage(pathData + parentReaction + "/" + imageList[imageIndx]))
				templabels.append(currentLabel)
				#outdata.append(parseImage(pathData + parentReaction + "/" + imageList[imageIndx]))
				#outlabels.append(currentLabel)									

			outdata.append(tempdata)
			outlabels.append(templabels)
	
	# equalize size of data for each emotion
	minsize = len(outdata[0])
	for dataset in outdata:
		if len(dataset) < minsize:
			minsize = len(dataset)
	for dataset in outdata:
		finaloutdata += dataset[:minsize]
	for dataset in outlabels:
		finaloutlabels += dataset[:minsize]
	

	return np.array(finaloutdata) / 255.0, np.array(finaloutlabels) / 255.0, emotionLabels
	'''
	for emotion in inputEmotions.iterkeys():
		del outdata[:]
		del outlabels[:]
		sumofemotions = 0
		for reaction in inputEmotions[emotion].iterkeys():
			sumofemotions += len(allData[reaction])

		for reaction in inputEmotions[emotion].iterkeys():
			wanted = inputEmotions[emotion][reaction]
			availabledata = len(allData[reaction])
			actualprop = availabledata/float(sumofemotions)
			maxdiff = 0
			maxemotion = ""
			if (wanted > actualprop):
				diff = wanted - actualprop
				print "test",reaction + str(diff)
				if diff > maxdiff:
					maxdiff = diff
					maxemotion = reaction


		if(maxemotion != ""): #the data needs adjusting to meet the requirements of proportion
			newTotal = len(allData[maxemotion]) / inputEmotions[emotion][maxemotion]
			for reaction in inputEmotions[emotion].iterkeys():
				eminstances = int(newTotal * inputEmotions[emotion][reaction])
				for i in xrange(eminstances):
					if not emotion in emotionLabels.keys():
						emotionLabels[emotion] = np.array(allData[emotion][0])
					outdata.append(allData[reaction][i])
					outlabels.append(allData[emotion][0])

					#outdata += allData[reaction][i]
					#outlabels += allData[emotion][0]

		else: #data does not need adjusting
			for emo in inputEmotions[emotion].iterkeys():
				if not emotion in emotionLabels.keys():
					emotionLabels[emotion] = np.array(allData[emotion][0])
				outdata += allData[emo]
				outlabels += [allData[emotion][0] for i in xrange(len(allData[emo]))]
				#outdata += allData[emo]
				#outlabels += [allData[emotion][0] for i in xrange(len(allData[emo]))]

		print "outdata",np.array(outdata).shape
		print "labels",type(outlabels)
		print "datashape",len(outlabels[0])


		out[emotion] = (np.array(outdata), np.array(outlabels))
	'''
	#return np.array(outdata), np.array(outlabels)


def multiplyData(input, degree):
	for key in input.iterkeys():
		print "type ",input[key][0].shape
		noisedDataset = []
		noisedLabels = []
		for d in xrange(degree-1):
			for image in xrange(input[key][0].shape[0]):
				imagecpy = copy.copy(input[key][0][image,:])
				labelcpy = copy.copy(input[key][1][image,:])
				for i in xrange(10):
					x = random.randint(0, len(imagecpy)-1)
					y = random.randint(0, len(imagecpy)-1)
					a = imagecpy[x]
					imagecpy[x] = imagecpy[y]
					imagecpy[y] = a
				noisedDataset.append(imagecpy)

				for i in xrange(10):
					x = random.randint(0, len(labelcpy)-1)
					y = random.randint(0, len(labelcpy)-1)
					a = labelcpy[x]
					labelcpy[x] = labelcpy[y]
					labelcpy[y] = a
				noisedLabels.append(labelcpy)	

		input[key] = (np.vstack((input[key][0], np.array(noisedDataset))), np.vstack((input[key][1], np.array(noisedLabels))))

	print "asdfas",input["happy"][0].shape
	print "asdfad",input["happy"][1].shape
	return input
'''
sadnessSecure = {"happy": 0.8, "anger":0.1, "sadness": 0.1}
happySecure = {"happy": 1}
childEmotions = {"happy": happySecure, "sadness": sadnessSecure}
readProportion(childEmotions)
'''