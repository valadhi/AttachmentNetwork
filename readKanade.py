import os.path
from PIL import Image 
import numpy as np
import cv2

imageSize = 25
#emotionArray = ["anger","contempt","disgust","fear","happy","sadness","surprise"]
#kanadeEmotions = {1:"anger", 2:"contempt", 3:"disgust", 4:"fear", 5:"happy", 6:"sadness", 7:"surprise"}

labelBits = imageSize**2
classLabels = {}
totHidden = 3
dataMultiplier = 2

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

def readEmotion(emo):
	out = []
	for fil in os.listdir("kanade/"+emo):
		out.append(parseImage("kanade/"+emo+"/"+fil))
	return np.array(out)
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
		path = "kanade/" + emote + "/"
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
'''
def read():
	global nrDataPoints, emotionArray, dataMultiplier,classLabels, imageSize, associations
	training_datapoints = (int)(0.7 * nrDataPoints)
	data = []
	labels = []
	for i in emotionArray:
		filename = "images/" + str(associations[i])+"-image("+str(imageSize)+", "+str(imageSize)+")0.jpg"
		#print filename
		classLabels[i] = parseImage(filename, i)
	    #classLabels[i] = putLabel(i)
	    #label for each emotion will be fixed to first image in associated
	    #categori
		for j in range(training_datapoints):
			imagefile = "images/" + str(i)+"-image("+str(imageSize)+", "+str(imageSize)+")"+str(j)+".jpg"
			#print imagefile
			parsedImg = parseImage(imagefile, i)
			for m in xrange(dataMultiplier):
				data.append(parsedImg)
				labels.append(classLabels[i])
	#print data
	#print classLabels[i].shape
	return np.array(data), np.array(labels)
'''