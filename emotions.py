""" The aim of this file is to contain all the function
and the main which have to do with emotion recognition, especially
with the Kanade database."""

import argparse
import cPickle as pickle
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix
import readKanade

import matplotlib.pyplot as plt
import numpy as np

import deepbelief as db
import restrictedBoltzmannMachine as rbm

from common import *
#from readfacedatabases import *
from activationfunctions import *


parser = argparse.ArgumentParser(description='emotion recongnition')
parser.add_argument('--rbmnesterov', dest='rbmnesterov',action='store_true', default=False,
                    help=("if true, rbms are trained using nesterov momentum"))
parser.add_argument('--save',dest='save',action='store_true', default=False,
                    help="if true, the network is serialized and saved")
parser.add_argument('--train',dest='train',action='store_true', default=False,
                    help=("if true, the network is trained from scratch from the"
                          "traning data"))
parser.add_argument('--rbm', dest='rbm',action='store_true', default=False,
                    help=("if true, the code for traning an rbm on the data is run"))
parser.add_argument('--sparsity', dest='sparsity',action='store_true', default=False,
                    help=("if true, the the networks are trained with sparsity constraints"))
parser.add_argument('--dbKanade', dest='dbKanade',action='store_true', default=False,
                    help=("if true, the code for training a deepbelief net on the"
                          "data is run, where the supervised data is the Kanade DB"))
parser.add_argument('--dbPIE', dest='dbPIE',action='store_true', default=False,
                    help=("if true, the code for training a deepbelief net on the"
                          "data is run, where the supervised data is the PIE DB"))
parser.add_argument('--trainSize', type=int, default=10000,
                    help='the number of tranining cases to be considered')
parser.add_argument('--testSize', type=int, default=1000,
                    help='the number of testing cases to be considered')
parser.add_argument('netFile', help="file where the serialized network should be saved")
parser.add_argument('--nesterov', dest='nesterov',action='store_true', default=False,
                    help=("if true, the deep belief net is trained using nesterov momentum"))
parser.add_argument('--rmsprop', dest='rmsprop',action='store_true', default=False,
                    help=("if true, rmsprop is used when training the deep belief net."))
parser.add_argument('--rbmrmsprop', dest='rbmrmsprop',action='store_true', default=False,
                    help=("if true, rmsprop is used when training the rbms."))
parser.add_argument('--cv', dest='cv',action='store_true', default=False,
                    help=("if true, do cross validation"))
parser.add_argument('--cvPIE', dest='cvPIE',action='store_true', default=False,
                    help=("if true, do cross validation"))
parser.add_argument('--svmPIE', dest='svmPIE',action='store_true', default=False,
                    help=("if true, do SVM on top of the last hidden features"))
parser.add_argument('--average', dest='average',action='store_true', default=False,
                    help=("average out results over multiple runs"))
parser.add_argument('--illumination',dest='illumination',action='store_true', default=False,
                    help="if true, trains and tests the images with different illuminations")
parser.add_argument('--pose',dest='pose',action='store_true', default=False,
                    help="if true, trains and tests the images with different poses")
parser.add_argument('--subjects',dest='subjects',action='store_true', default=False,
                    help="if true, trains and tests the images with different subjects")
parser.add_argument('--missing', dest='missing',action='store_true', default=False,
                    help=("tests the network with missing data."))
parser.add_argument('--crossdb', dest='crossdb',action='store_true', default=False,
                    help=("if true, trains the DBN with multi pie and tests with Kanade."))
parser.add_argument('--crossdbCV', dest='crossdbCV',action='store_true', default=False,
                    help=("if true, trains the DBN with multi pie and tests with Kanade."))
parser.add_argument('--facedetection', dest='facedetection',action='store_true', default=False,
                    help=("if true, do face detection"))
parser.add_argument('--maxEpochs', type=int, default=1000,
                    help='the maximum number of supervised epochs')
parser.add_argument('--miniBatchSize', type=int, default=10,
                    help='the number of training points in a mini batch')
parser.add_argument('--validation',dest='validation',action='store_true', default=False,
                    help="if true, the network is trained using a validation set")
parser.add_argument('--equalize',dest='equalize',action='store_true', default=False,
                    help="if true, the input images are equalized before being fed into the net")
parser.add_argument('--crop',dest='crop',action='store_true', default=False,
                    help="crops images from databases before training the net")
parser.add_argument('--relu', dest='relu',action='store_true', default=False,
                    help=("if true, trains the RBM or DBN with a rectified linear unit"))
parser.add_argument('--preTrainEpochs', type=int, default=1,
                    help='the number of pretraining epochs')
parser.add_argument('--kaggle',dest='kaggle',action='store_true', default=False,
                      help='if true, trains a net on the kaggle data')


# DEBUG mode?
parser.add_argument('--debug', dest='debug',action='store_false', default=False,
                    help=("if true, the deep belief net is ran in DEBUG mode"))

# Get the arguments of the program
args = parser.parse_args()

# Set the debug mode in the deep belief net
db.DEBUG = args.debug

SMALL_SIZE = ((50, 25))
def learnEmotions(emotions):
  for e in emotions:
    data = readKanade.readEmotion(e)
    data = data/255
    np.random.shuffle(data)
    nrVisible = len(data[0])
    nrHidden = 800
    activationFunction = Sigmoid()
    net = rbm.RBM(nrVisible, nrHidden, 1.4, 0.75, 0.75,
              visibleActivationFunction=activationFunction,
              hiddenActivationFunction=activationFunction,
              rmsprop=args.rbmrmsprop,
              nesterov=args.rbmnesterov,
              sparsityConstraint=args.sparsity,
              sparsityRegularization=0.5,
              trainingEpochs=args.maxEpochs,
              sparsityTraget=0.01,
              fixedLabel = False)
    net.train(data)
    rando = np.random.random_sample((25,25))
    print "data",data.shape[1]
    print "rando",rando.shape
    rando = rando.reshape(1, data.shape[1])

    recon = net.reconstruct(rando, 50)
    print "rando",rando.shape
    saveImage(recon, e, (25,25))

    t = visualizeWeights(net.weights.T, (25,25), (10,10)) 
    plt.imshow(t, cmap=plt.cm.gray)
    plt.axis('off')

    plt.savefig('weights' + e + '.png', transparent=True) 

def rbmEmotions(big=False, reconstructRandom=False):
  #data, labels = readMultiPIE(big, equalize=args.equalize)

  emotions = ["fear", "happy", "anger", "contempt", "disgust", "sadness", "surprise"]
  asoc = {"fear": "happy", "anger":"fear", "happy":"anger"}
  learnEmotions(emotions)
  '''
  data, labels = readKanade.readWithAsoc(asoc)

  print "data.shape"
  print data.shape
  print "labels.shape",labels.shape
  data = data / 255.0
  labels = labels / 255.0

  label1 = labels[0,:]
  label2 = labels[int(labels.shape[0]/3),:]
  label3 = labels[int(labels.shape[0]-1),:]

  if args.relu:
    activationFunction = Rectified()
    data = scale(data)
  else:
    activationFunction = Sigmoid()

  #trainData = data[0:-1, :]
  Data = np.concatenate((data, labels), axis=1)
  np.random.shuffle(Data)
  trainData = Data[0:-1, :]
  print "trainData",trainData.shape
  # Train the network
  if args.train:
    # The number of hidden units is taken from a deep learning tutorial
    # The data are the values of the images have to be normalized before being
    # presented to the network
    nrVisible = len(trainData[0])
    nrHidden = 800
    # use 1 dropout to test the rbm for now
    net = rbm.RBM(nrVisible, nrHidden, 0.1, 1, 1,
                  visibleActivationFunction=activationFunction,
                  hiddenActivationFunction=activationFunction,
                  rmsprop=args.rbmrmsprop,
                  nesterov=args.rbmnesterov,
                  sparsityConstraint=args.sparsity,
                  sparsityRegularization=0.5,
                  trainingEpochs=args.maxEpochs,
                  sparsityTraget=0.01,
                  fixedLabel = True)
    net.train(trainData)
    print net.weights.T.shape
    t = visualizeWeights(net.weights.T, SMALL_SIZE, (10,10))
  else:
    # Take the saved network and use that for reconstructions
    f = open(args.netFile, "rb")
    t = pickle.load(f)
    net = pickle.load(f)
    f.close()

  # get a random image and see it looks like
  # if reconstructRandom:
  #   test = np.random.random_sample(test.shape)

  # Show the initial image first
  test = Data[-1, :]
  print "test.shape"
  print test.shape

  plt.imshow(vectorToImage(test, SMALL_SIZE), cmap=plt.cm.gray)
  plt.axis('off')
  plt.savefig('initialface.png', transparent=True)

  rando = np.random.random_sample((25,25))
  rando = rando.reshape(1, label1.shape[0])
  label1 = label1.reshape(1, label1.shape[0])
  label2 = label2.reshape(1, label2.shape[0])
  label3 = label3.reshape(1, label3.shape[0])
  print "rando",rando.shape
  print "label1",label1.shape
  print "label2",label2.shape
  print "label3",label3.shape
  recon1 = np.concatenate((rando, label1), axis=1)
  recon2 = np.concatenate((rando, label2), axis=1)
  recon3 = np.concatenate((rando, label3), axis=1)
  print "recon1",recon1.shape
  #recon1 = recon1.reshape(1, recon1.shape[0])
  recon1 = net.reconstruct(recon1)
  recon2 = net.reconstruct(recon2)
  recon3 = net.reconstruct(recon3)
  saveImage(recon1, "recon1")
  saveImage(recon2, "recon2")
  saveImage(recon3, "recon3")

  recon = net.reconstruct(test.reshape(1, test.shape[0]))
  print recon.shape

  plt.imshow(vectorToImage(recon, SMALL_SIZE), cmap=plt.cm.gray)
  plt.axis('off')
  plt.savefig('reconstructface.png', transparent=True)

  #reconrndm = net.reconstruct(rando.reshape(1, rando.shape[0]),50)
  #saveImage(reconrndm, "randomGen")

  # Show the weights and their form in a tile fashion
  # Plot the weights
  plt.imshow(t, cmap=plt.cm.gray)
  plt.axis('off')
  if args.rbmrmsprop:
    st='rmsprop'
  else:
    st = 'simple'
  plt.savefig('weights' + st + '.png', transparent=True)


  # let's make some sparsity checks
  hidden = net.hiddenRepresentation(test.reshape(1, test.shape[0]))
  print hidden.sum()
  print "done"


  if args.save:
    f = open(args.netFile, "wb")
    pickle.dump(t, f)
    pickle.dump(net, f)
'''

def saveImage(data, name, size):
  plt.imshow(vectorToImage(data, size), cmap=plt.cm.gray)
  plt.axis('off')
  plt.savefig("dump/"+name + '.png',transparent=True) 

"""
  Arguments:
    big: should the big or small images be used?
"""
def deepbeliefKanadeCV(big=False):
  data, labels = readKanade(big, None, equalize=args.equalize)

  data, labels = shuffle(data, labels)

  print "data.shape"
  print data.shape
  print "labels.shape"
  print labels.shape

  if args.relu:
    activationFunction = Rectified()
    rbmActivationFunctionVisible = Identity()
    rbmActivationFunctionHidden = RectifiedNoisy()
    data = scale(data)
  else:
    activationFunction = Sigmoid()
    rbmActivationFunctionVisible = Sigmoid()
    rbmActivationFunctionHidden = Sigmoid()


  # TODO: try boosting for CV in order to increase the number of folds
  params =[(0.1, 0.1, 0.9), (0.1,  0.5, 0.9),  (0.5, 0.1, 0.9),  (0.5, 0.5, 0.9),
           (0.1, 0.1, 0.95), (0.1, 0.5, 0.95), (0.5, 0.1, 0.95), (0.5, 0.5, 0.95),
           (0.1, 0.1, 0.99), (0.1, 0.5, 0.99), (0.5, 0.1, 0.99), (0.5, 0.5, 0.99)]

  unsupervisedData = buildUnsupervisedDataSetForKanadeLabelled()
  # print "unsupervisedData.shape"
  # print unsupervisedData.shape

  kf = cross_validation.KFold(n=len(data), k=len(params))
  bestCorrect = 0
  bestProbs = 0

  fold = 0
  for train, test in kf:

    trainData = data[train]
    trainLabels = labels[train]

    # TODO: this might require more thought
    net = db.DBN(5, [1200, 1800, 1800, 1800, 7],
               binary=1-args.relu,
               activationFunction=activationFunction,
               rbmActivationFunctionVisible=rbmActivationFunctionVisible,
               rbmActivationFunctionHidden=rbmActivationFunctionHidden,
               unsupervisedLearningRate=params[fold][0],
               supervisedLearningRate=params[fold][1],
               momentumMax=params[fold][2],
               nesterovMomentum=args.nesterov,
               rbmNesterovMomentum=args.rbmnesterov,
               rmsprop=args.rmsprop,
               miniBatchSize=args.miniBatchSize,
               hiddenDropout=0.5,
               visibleDropout=0.8,
               rbmHiddenDropout=1.0,
               rbmVisibleDropout=1.0,
               preTrainEpochs=args.preTrainEpochs)

    net.train(trainData, trainLabels,
              maxEpochs=args.maxEpochs,
              validation=args.validation,
              unsupervisedData=unsupervisedData)

    probs, predicted = net.classify(data[test])

    actualLabels = labels[test]
    correct = 0
    errorCases = []

    for i in xrange(len(test)):
      print "predicted"
      print "probs"
      print probs[i]
      print predicted[i]
      print "actual"
      actual = actualLabels[i]
      print np.argmax(actual)
      if predicted[i] == np.argmax(actual):
        correct += 1
      else:
        errorCases.append(i)

    print "correct for " + str(params[fold])
    print correct

    if bestCorrect < correct:
      bestCorrect = correct
      bestParam = params[fold]
      bestProbs = correct * 1.0 / len(test)

    fold += 1

  print "bestParam"
  print bestParam

  print "bestProbs"
  print bestProbs


def deepbeliefKanade(big=False):
  data, labels = readKanade(big, None, equalize=args.equalize)

  data, labels = shuffle(data, labels)

  print "data.shape"
  print data.shape
  print "labels.shape"
  print labels.shape

  # Random data for training and testing
  kf = cross_validation.KFold(n=len(data), n_folds=5)
  for train, test in kf:
    break

  if args.relu:
    activationFunction = Rectified()
    unsupervisedLearningRate = 0.05
    supervisedLearningRate = 0.01
    momentumMax = 0.95
    data = scale(data)
    rbmActivationFunctionVisible = Identity()
    rbmActivationFunctionHidden = RectifiedNoisy()

  else:
    activationFunction = Sigmoid()
    rbmActivationFunctionVisible = Sigmoid()
    rbmActivationFunctionHidden = Sigmoid()

    unsupervisedLearningRate = 0.5
    supervisedLearningRate = 0.1
    momentumMax = 0.9

  trainData = data[train]
  trainLabels = labels[train]

  # TODO: this might require more thought
  net = db.DBN(5, [1200, 1500, 1500, 1500, 7],
             binary=1-args.relu,
             activationFunction=activationFunction,
             rbmActivationFunctionVisible=rbmActivationFunctionVisible,
             rbmActivationFunctionHidden=rbmActivationFunctionHidden,
             unsupervisedLearningRate=unsupervisedLearningRate,
             supervisedLearningRate=supervisedLearningRate,
             momentumMax=momentumMax,
             nesterovMomentum=args.nesterov,
             rbmNesterovMomentum=args.rbmnesterov,
             rmsprop=args.rmsprop,
             miniBatchSize=args.miniBatchSize,
             hiddenDropout=0.5,
             visibleDropout=0.8,
             rbmVisibleDropout=1.0,
             rbmHiddenDropout=1.0,
             preTrainEpochs=args.preTrainEpochs)

  # unsupervisedData = buildUnsupervisedDataSetForKanadeLabelled()
  unsupervisedData = None
  # print "unsupervisedData.shape"
  # print unsupervisedData.shape

  net.train(trainData, trainLabels, maxEpochs=args.maxEpochs,
            validation=args.validation,
            unsupervisedData=unsupervisedData)

  probs, predicted = net.classify(data[test])

  actualLabels = labels[test]
  correct = 0
  errorCases = []

  for i in xrange(len(test)):
    print "predicted"
    print "probs"
    print probs[i]
    print "predicted"
    print predicted[i]
    print "actual"
    actual = actualLabels[i]
    print np.argmax(actual)
    if predicted[i] == np.argmax(actual):
      correct += 1
    else:
      errorCases.append(i)

  print "correct"
  print correct

  print "percentage correct"
  print correct  * 1.0/ len(test)

  confMatrix = confusion_matrix(np.argmax(actualLabels, axis=1), predicted)
  print "confusion matrix"
  print confMatrix

  if args.save:
    with open(args.netFile, "wb") as f:
      pickle.dump(net, f)


def buildUnsupervisedDataSetForKanadeLabelled():
  return readJaffe(args.crop, args.facedetection, equalize=args.equalize)
  # return np.vstack((readAttData(equalize=args.equalize),
                      # readCroppedYale
  #                   readJaffe(args.facedetection, equalize=args.equalize)))
                    # readAberdeen(args.crop, args.facedetection, equalize=args.equalize)))
    # readNottingham(),
    # readCroppedYale(),
    # readMultiPIE(equalize=args.equalize)[0]))

def buildUnsupervisedDataSetForPIE():
  return None

# TODO: you need to be able to map the emotions between each other
# but it might be the case that you won't get higher results which such a big
#dataset
def buildSupervisedDataSet():
  dataKanade, labelsKanade = readKanade(equalize=args.equalize)
  dataMPie, labelsMPie = readMultiPIE(equalize=args.equalize)
  print dataMPie.shape
  print dataKanade.shape

  data = np.vstack((dataKanade, dataMPie))
  labels = labelsKanade + labelsMPie
  return data, labels

def deepbeliefMultiPIE(big=False):
  data, labels = readMultiPIE(equalize=args.equalize)

  data, labels = shuffle(data, labels)

  print "data.shape"
  print data.shape
  print "labels.shape"
  print labels.shape

  # Random data for training and testing
  kf = cross_validation.KFold(n=len(data), n_folds=5)
  for train, test in kf:
    break

  if args.relu:
    activationFunction = RectifiedNoisy()
    rbmActivationFunctionHidden = RectifiedNoisy()
    rbmActivationFunctionVisible = Identity()
    unsupervisedLearningRate = 0.005
    supervisedLearningRate = 0.001
    momentumMax = 0.95
    data = scale(data)
  else:
    activationFunction = Sigmoid()
    rbmActivationFunctionHidden = Sigmoid()
    rbmActivationFunctionVisible = Sigmoid()
    unsupervisedLearningRate = 0.05
    supervisedLearningRate = 0.01
    momentumMax = 0.95

  trainData = data[train]
  trainLabels = labels[train]

  if args.train:
    # TODO: this might require more thought
    net = db.DBN(5, [1200, 1500, 1500, 1500, 6],
               binary=1-args.relu,
               activationFunction=activationFunction,
               rbmActivationFunctionVisible=rbmActivationFunctionVisible,
               rbmActivationFunctionHidden=rbmActivationFunctionHidden,
               unsupervisedLearningRate=unsupervisedLearningRate,
               supervisedLearningRate=supervisedLearningRate,
               momentumMax=momentumMax,
               nesterovMomentum=args.nesterov,
               rbmNesterovMomentum=args.rbmnesterov,
               rmsprop=args.rmsprop,
               miniBatchSize=args.miniBatchSize,
               visibleDropout=0.8,
               hiddenDropout=0.5,
               rbmHiddenDropout=1.0,
               rbmVisibleDropout=1.0,
               preTrainEpochs=args.preTrainEpochs)

    unsupervisedData = buildUnsupervisedDataSetForPIE()

    net.train(trainData, trainLabels, maxEpochs=args.maxEpochs,
              validation=args.validation,
              unsupervisedData=unsupervisedData)
  else:
     # Take the saved network and use that for reconstructions
    with open(args.netFile, "rb") as f:
      net = pickle.load(f)

  probs, predicted = net.classify(data[test])

  actualLabels = labels[test]
  correct = 0
  errorCases = []

  for i in xrange(len(test)):
    print "predicted"
    print "probs"
    print probs[i]
    print "predicted"
    print predicted[i]
    print "actual"
    actual = actualLabels[i]
    print np.argmax(actual)
    if predicted[i] == np.argmax(actual):
      correct += 1
    else:
      errorCases.append(i)

  print "correct"
  print correct

  print "percentage correct"
  print correct  * 1.0/ len(test)

  print type(predicted)
  print type(actualLabels)
  print predicted.shape
  print actualLabels.shape

  confMatrix = confusion_matrix(np.argmax(actualLabels, axis=1), predicted)

  print "confusion matrix"
  print confMatrix

  if args.save:
    with open(args.netFile, "wb") as f:
      pickle.dump(net, f)


def deepbeliefMultiPIEAverage(big=False):
  data, labels = readMultiPIE(equalize=args.equalize)

  data, labels = shuffle(data, labels)

  print "data.shape"
  print data.shape
  print "labels.shape"
  print labels.shape

  if args.relu:
    activationFunction = Rectified()
    rbmActivationFunctionHidden = RectifiedNoisy()
    rbmActivationFunctionVisible = identityIdentity()
    unsupervisedLearningRate = 0.005
    supervisedLearningRate = 0.001
    momentumMax = 0.95
    data = scale(data)
  else:
    activationFunction = Sigmoid()
    rbmActivationFunctionHidden = Sigmoid()
    rbmActivationFunctionVisible = Sigmoid()
    unsupervisedLearningRate = 0.05
    supervisedLearningRate = 0.01
    momentumMax = 0.95


  correctAll = []
  confustionMatrices = []
  # Random data for training and testing
  kf = cross_validation.KFold(n=len(data), n_folds=5)
  for train, test in kf:


    trainData = data[train]
    trainLabels = labels[train]

    if args.train:
      # TODO: this might require more thought
      net = db.DBN(5, [1200, 1500, 1500, 1500, 6],
                 binary=1-args.relu,
                 activationFunction=activationFunction,
                 rbmActivationFunctionVisible=rbmActivationFunctionVisible,
                 rbmActivationFunctionHidden=rbmActivationFunctionHidden,
                 unsupervisedLearningRate=unsupervisedLearningRate,
                 supervisedLearningRate=supervisedLearningRate,
                 momentumMax=momentumMax,
                 nesterovMomentum=args.nesterov,
                 rbmNesterovMomentum=args.rbmnesterov,
                 rmsprop=args.rmsprop,
                 miniBatchSize=args.miniBatchSize,
                 visibleDropout=0.8,
                 hiddenDropout=0.5,
                 rbmHiddenDropout=1.0,
                 rbmVisibleDropout=1.0,
                 preTrainEpochs=args.preTrainEpochs)

      unsupervisedData = buildUnsupervisedDataSetForPIE()

      net.train(trainData, trainLabels, maxEpochs=args.maxEpochs,
                validation=args.validation,
                unsupervisedData=unsupervisedData)
    else:
       # Take the saved network and use that for reconstructions
      with open(args.netFile, "rb") as f:
        net = pickle.load(f)

    probs, predicted = net.classify(data[test])

    actualLabels = labels[test]
    correct = 0
    errorCases = []

    for i in xrange(len(test)):
      actual = actualLabels[i]
      print np.argmax(actual)
      if predicted[i] == np.argmax(actual):
        correct += 1
      else:
        errorCases.append(i)

    print "correct"
    print correct

    print "percentage correct"
    print correct  * 1.0/ len(test)

    print type(predicted)
    print type(actualLabels)
    print predicted.shape
    print actualLabels.shape

    confMatrix = confusion_matrix(np.argmax(actualLabels, axis=1), predicted)

    print "confusion matrix"
    print confMatrix

    correctAll += [correct  * 1.0/ len(test)]
    confustionMatrices += [confMatrix]

  print "average correct"
  print sum(correctAll) / len(correctAll)
  print "average confusion matrix"
  print sum(confustionMatrices) * 1.0 / len(confustionMatrices)


def deepbeliefPIECV(big=False):
  data, labels = readMultiPIE(equalize=args.equalize)

  data, labels = shuffle(data, labels)

  print "data.shape"
  print data.shape
  print "labels.shape"
  print labels.shape

  if args.relu:
    activationFunction = Rectified()
    rbmActivationFunctionVisible = Identity()
    rbmActivationFunctionHidden = RectifiedNoisy()
    # IMPORTANT: SCALE THE DATA IF YOU USE GAUSSIAN VISIBlE UNITS
    data = scale(data)
  else:
    activationFunction = Sigmoid()
    rbmActivationFunctionVisible = Sigmoid()
    rbmActivationFunctionHidden = Sigmoid()

  # TODO: try boosting for CV in order to increase the number of folds
  # params =[ (0.01, 0.05, 0.9),  (0.05, 0.01, 0.9),  (0.05, 0.05, 0.9),
  #           (0.01, 0.05, 0.95), (0.05, 0.01, 0.95), (0.05, 0.05, 0.95),
  #           (0.01, 0.05, 0.99), (0.05, 0.01, 0.99), (0.05, 0.05, 0.99)]


  # params =[(0.05, 0.01, 0.9,  0.8, 1.0), (0.05, 0.01, 0.9,  1.0, 1.0), (0.05, 0.01, 0.9,  0.8, 0.5), (0.05, 0.01, 0.9, 1.0, 0.5),
  #          (0.05, 0.01, 0.95, 0.8, 1.0), (0.05, 0.01, 0.95, 1.0, 1.0), (0.05, 0.01, 0.95, 0.8, 0.5), (0.05, 0.01, 0.95, 1.0, 0.5),
  #          (0.05, 0.01, 0.99, 0.8, 1.0), (0.05, 0.01, 0.99, 1.0, 1.0), (0.05, 0.01, 0.99, 0.8, 0.5), (0.05, 0.01, 0.99, 1.0, 0.5)]

  if args.relu:
    params =[(0.005, 0.001, 0.95,  0.8, 1.0), (0.05, 0.01, 0.95,  0.8, 1.0), (0.005, 0.01, 0.95,  0.8, 1.0), (0.05, 0.001, 0.95,  0.8, 1.0)]
  else:
    params =[(0.05, 0.01, 0.95,  0.8, 0.8), (0.05, 0.01, 0.95,  0.8, 1.0), (0.05, 0.01, 0.95,  1.0, 1.0), (0.05, 0.01, 0.95,  0.8, 0.5), (0.05, 0.01, 0.95, 1.0, 0.5)]

  #          (0.05, 0.01, 0.95, 0.8, 1.0), (0.05, 0.01, 0.95, 1.0, 1.0), (0.05, 0.01, 0.95, 0.8, 0.5), (0.05, 0.01, 0.95, 1.0, 0.5),
  #          (0.05, 0.01, 0.99, 0.8, 1.0), (0.05, 0.01, 0.99, 1.0, 1.0), (0.05, 0.01, 0.99, 0.8, 0.5), (0.05, 0.01, 0.99, 1.0, 0.5)]


  unsupervisedData = buildUnsupervisedDataSetForPIE()

  kf = cross_validation.KFold(n=len(data), k=len(params))
  bestCorrect = 0
  bestProbs = 0

  probsforParms = []
  fold = 0
  for train, test in kf:

    trainData = data[train]
    trainLabels = labels[train]

    # TODO: this might require more thought
    net = db.DBN(5, [1200, 1500, 1500, 1500, 6],
               binary=1-args.relu,
               activationFunction=activationFunction,
               rbmActivationFunctionVisible=rbmActivationFunctionVisible,
               rbmActivationFunctionHidden=rbmActivationFunctionHidden,
               unsupervisedLearningRate=params[fold][0],
               supervisedLearningRate=params[fold][1],
               momentumMax=params[fold][2],
               nesterovMomentum=args.nesterov,
               rbmNesterovMomentum=args.rbmnesterov,
               rmsprop=args.rmsprop,
               miniBatchSize=args.miniBatchSize,
               rbmHiddenDropout=1.0,
               rbmVisibleDropout=1.0,
               visibleDropout=params[fold][3],
               hiddenDropout=params[fold][4],
               preTrainEpochs=args.preTrainEpochs)

    net.train(trainData, trainLabels,
              maxEpochs=args.maxEpochs,
              validation=args.validation,
              unsupervisedData=unsupervisedData)

    probs, predicted = net.classify(data[test])

    actualLabels = labels[test]
    correct = 0
    errorCases = []

    for i in xrange(len(test)):
      print "predicted"
      print "probs"
      print probs[i]
      print predicted[i]
      print "actual"
      actual = actualLabels[i]
      print np.argmax(actual)
      if predicted[i] == np.argmax(actual):
        correct += 1
      else:
        errorCases.append(i)

    print "correct for " + str(params[fold])
    print correct
    print "correctProbs"
    correctProbs = correct * 1.0 / len(test)
    print correctProbs

    probsforParms += [correctProbs]

    if bestCorrect < correct:
      bestCorrect = correct
      bestParam = params[fold]
      bestProbs = correctProbs
    fold += 1

  print "bestParam"
  print bestParam

  print "bestProbs"
  print bestProbs


  for i in xrange(len(params)):
    print "parameter tuple " + str(params[i]) + " achieved correctness of " + str(probsforParms[i])

def deepbeliefKaggleCompetition(big=False):
  data, labels = readKaggleCompetition()

  data, labels = shuffle(data, labels)

  print "data.shape"
  print data.shape
  print "labels.shape"
  print labels.shape

  # Random data for training and testing
  kf = cross_validation.KFold(n=len(data), n_folds=5)
  for train, test in kf:
    break

  if args.relu:
    activationFunction = Rectified()
    unsupervisedLearningRate = 0.05
    supervisedLearningRate = 0.01
    momentumMax = 0.95
    data = scale(data)
    rbmActivationFunctionVisible = Identity()
    rbmActivationFunctionHidden = RectifiedNoisy()

  else:
    activationFunction = Sigmoid()
    rbmActivationFunctionVisible = Sigmoid()
    rbmActivationFunctionHidden = Sigmoid()

    unsupervisedLearningRate = 0.5
    supervisedLearningRate = 0.1
    momentumMax = 0.9

  trainData = data[train]
  trainLabels = labels[train]

  # TODO: this might require more thought
  net = db.DBN(5, [2304, 1500, 1500, 1500, 7],
             binary=1-args.relu,
             activationFunction=activationFunction,
             rbmActivationFunctionVisible=rbmActivationFunctionVisible,
             rbmActivationFunctionHidden=rbmActivationFunctionHidden,
             unsupervisedLearningRate=unsupervisedLearningRate,
             supervisedLearningRate=supervisedLearningRate,
             momentumMax=momentumMax,
             nesterovMomentum=args.nesterov,
             rbmNesterovMomentum=args.rbmnesterov,
             rmsprop=args.rmsprop,
             miniBatchSize=args.miniBatchSize,
             hiddenDropout=0.5,
             visibleDropout=0.8,
             rbmVisibleDropout=1.0,
             rbmHiddenDropout=1.0,
             preTrainEpochs=args.preTrainEpochs)

  # unsupervisedData = buildUnsupervisedDataSetForKanadeLabelled()
  unsupervisedData = None
  # print "unsupervisedData.shape"
  # print unsupervisedData.shape

  net.train(trainData, trainLabels, maxEpochs=args.maxEpochs,
            validation=args.validation,
            unsupervisedData=unsupervisedData)

  probs, predicted = net.classify(data[test])

  actualLabels = labels[test]
  correct = 0
  errorCases = []

  for i in xrange(len(test)):
    print "predicted"
    print "probs"
    print probs[i]
    print "predicted"
    print predicted[i]
    print "actual"
    actual = actualLabels[i]
    print np.argmax(actual)
    if predicted[i] == np.argmax(actual):
      correct += 1
    else:
      errorCases.append(i)

  print "correct"
  print correct

  print "percentage correct"
  print correct  * 1.0/ len(test)

  confMatrix = confusion_matrix(np.argmax(actualLabels, axis=1), predicted)
  print "confusion matrix"
  print confMatrix

  if args.save:
    with open(args.netFile, "wb") as f:
      pickle.dump(net, f)


def svmPIE():
  with open(args.netFile, "rb") as f:
    dbnNet = pickle.load(f)

  data, labels = readMultiPIE(equalize=args.equalize)

  # Random data for training and testing
  kf = cross_validation.KFold(n=len(data), n_folds=5)
  for train, test in kf:
    break

  training = data[train]
  trainLabels = labels[train]

  testing = data[test]
  testLabels = labels[test]

  svm.SVMCV(dbnNet, training, trainLabels, testing, testLabels)


# Make this more general to be able
# to say different subjects and different poses
# I tihnk the different subjects is very intersting
# and I should do this for for
def deepBeliefPieDifferentConditions():

  if args.illumination:
    getDataFunction = readMultiPieDifferentIlluminations
    allConditions = np.array(range(5))
  elif args.pose:
    getDataFunction = readMultiPieDifferentPoses
    allConditions = np.array(range(5))
  elif args.subjects:
    getDataFunction = readMultiPieDifferentSubjects
    allConditions = np.array(range(147))


  kf = cross_validation.KFold(n=len(allConditions), k=5)

  confustionMatrices = []
  correctAll = []

  for trainConditions, _ in kf:
    print "trainConditions"
    print trainConditions
    print trainConditions.shape

    trainData, trainLabels, testData, testLabels = getDataFunction(trainConditions, equalize=args.equalize)

    trainData, trainLabels = shuffle(trainData, trainLabels)

    print "input shape"
    print trainData[0].shape
    print "type(trainData)"
    print type(trainData)

    if args.relu:
      activationFunction = Rectified()
      rbmActivationFunctionVisible = Identity()
      rbmActivationFunctionHidden = RectifiedNoisy()

      unsupervisedLearningRate = 0.005
      supervisedLearningRate = 0.001
      momentumMax = 0.95
      trainData = scale(trainData)
      testData = scale(testData)

    else:
      activationFunction = Sigmoid()
      rbmActivationFunctionVisible = Sigmoid()
      rbmActivationFunctionHidden = Sigmoid()

      unsupervisedLearningRate = 0.05
      supervisedLearningRate = 0.01
      momentumMax = 0.9

    if args.train:
      # TODO: this might require more thought
      net = db.DBN(5, [1200, 1500, 1500, 1500, 6],
                 binary=1-args.relu,
                 activationFunction=activationFunction,
                 rbmActivationFunctionVisible=rbmActivationFunctionVisible,
                 rbmActivationFunctionHidden=rbmActivationFunctionHidden,
                 unsupervisedLearningRate=unsupervisedLearningRate,
                 # is this not a bad learning rate?
                 supervisedLearningRate=supervisedLearningRate,
                 momentumMax=momentumMax,
                 nesterovMomentum=args.nesterov,
                 rbmNesterovMomentum=args.rbmnesterov,
                 rmsprop=args.rmsprop,
                 miniBatchSize=args.miniBatchSize,
                 visibleDropout=0.8,
                 hiddenDropout=1.0,
                 rbmHiddenDropout=1.0,
                 rbmVisibleDropout=1.0,
                 preTrainEpochs=args.preTrainEpochs)

      print "trainData.shape"
      print trainData.shape
      net.train(trainData, trainLabels, maxEpochs=args.maxEpochs,
                validation=args.validation,
                unsupervisedData=None)
    else:
       # Take the saved network and use that for reconstructions
      with open(args.netFile, "rb") as f:
        net = pickle.load(f)

    probs, predicted = net.classify(testData)

    actualLabels = testLabels
    correct = 0
    errorCases = []

    for i in xrange(len(testLabels)):
      print "predicted"
      print "probs"
      print probs[i]
      print "predicted"
      print predicted[i]
      print "actual"
      actual = actualLabels[i]
      print np.argmax(actual)
      if predicted[i] == np.argmax(actual):
        correct += 1
      else:
        errorCases.append(i)

    print "correct"
    print correct

    print "percentage correct"
    correct = correct  * 1.0/ len(testLabels)
    print correct

    confMatrix = confusion_matrix(np.argmax(actualLabels, axis=1), predicted)

    print "confusion matrix"
    print confMatrix

    confustionMatrices += [confMatrix]
    correctAll += [correct]


  for i in allConditions:
    print "for condition" + str(i)
    print "the correct rate was " + str(correctAll[i])
    print "the confusionMatrix was " + str(confustionMatrices[i])

  print "average correct rate", sum(correctAll) * 1.0 / len(correctAll)
  print "average confusionMatrix was ", sum(confustionMatrices) * 1.0 / len(confustionMatrices)



# TODO: try with the same poses, it will work bad with training with all poses I think
"""Train with PIE test with Kanade. Check the equalization code. """
# TODO: try to add some unsupervised data
def crossDataBase():
  # Only train with the frontal pose
  trainData, trainLabels, _, _ = readMultiPieDifferentPoses([2], equalize=args.equalize)
  trainData, trainLabels = shuffle(trainData, trainLabels)

  print "trainLabels"
  print np.argmax(trainLabels, axis=1)

  # for i in xrange(len(trainLabels)):
  #   print "emotions", np.argmax(trainLabels[i])
  #   plt.imshow(vectorToImage(trainData[i], SMALL_SIZE), cmap=plt.cm.gray)
  #   plt.show()


  testData, testLabels = readKanade(False, None, equalize=args.equalize, vectorizeLabels=False)
  print "testLabels"
  print testLabels
  # Some emotions do not correspond for a to b, so we have to map them
  testData, testLabels = mapKanadeToPIELabels(testData, testLabels)
  testLabels = labelsToVectors(testLabels, 6)

  print "testLabels after map"
  labelsSimple = np.argmax(testLabels, axis=1)
  print labelsSimple

  # for i in xrange(len(labelsSimple)):
  #   print "emotions", labelsSimple[i]
  #   plt.imshow(vectorToImage(testData[i], SMALL_SIZE), cmap=plt.cm.gray)
  #   plt.show()


  if args.relu:
    activationFunction = Rectified() # Now I can even use rectifiednoisy because I use the deterministic version
    rbmActivationFunctionHidden = RectifiedNoisy()
    rbmActivationFunctionVisible = Identity()

    unsupervisedLearningRate = 0.005
    supervisedLearningRate = 0.001
    momentumMax = 0.95
    trainData = scale(trainData)
    testData = scale(testData)
  else:
    activationFunction = Sigmoid()
    rbmActivationFunctionHidden = Sigmoid()
    rbmActivationFunctionVisible = Sigmoid()
    unsupervisedLearningRate = 0.05
    supervisedLearningRate = 0.01
    momentumMax = 0.95


  if args.train:
    # TODO: this might require more thought
    net = db.DBN(5, [1200, 1500, 1500, 1500, 6],
               binary=1-args.relu,
               activationFunction=activationFunction,
               rbmActivationFunctionVisible=rbmActivationFunctionVisible,
               rbmActivationFunctionHidden=rbmActivationFunctionHidden,
               unsupervisedLearningRate=unsupervisedLearningRate,
               supervisedLearningRate=supervisedLearningRate,
               momentumMax=momentumMax,
               nesterovMomentum=args.nesterov,
               rbmNesterovMomentum=args.rbmnesterov,
               rmsprop=args.rmsprop,
               miniBatchSize=args.miniBatchSize,
               visibleDropout=0.8,
               hiddenDropout=1.0,
               rbmHiddenDropout=1.0,
               rbmVisibleDropout=1.0,
               preTrainEpochs=args.preTrainEpochs)

    unsupervisedData = buildUnsupervisedDataSetForPIE()

    net.train(trainData, trainLabels, maxEpochs=args.maxEpochs,
              validation=args.validation,
              unsupervisedData=unsupervisedData)

    if args.save:
      with open(args.netFile, "wb") as f:
        pickle.dump(net, f)

  else:
     # Take the saved network and use that for reconstructions
    with open(args.netFile, "rb") as f:
      net = pickle.load(f)

  probs, predicted = net.classify(testData)

  actualLabels = testLabels
  correct = 0
  errorCases = []

  for i in xrange(len(testLabels)):
    print "predicted"
    print "probs"
    print probs[i]
    print "predicted"
    print predicted[i]
    print "actual"
    actual = actualLabels[i]
    print np.argmax(actual)
    if predicted[i] == np.argmax(actual):
      correct += 1
    else:
      errorCases.append(i)

  print "correct"
  print correct

  print "percentage correct"
  print correct  * 1.0/ len(testLabels)

  print type(predicted)
  print type(actualLabels)
  print predicted.shape
  print actualLabels.shape

  confMatrix = confusion_matrix(np.argmax(actualLabels, axis=1), predicted)

  print "confusion matrix"
  print confMatrix

# TODO: try with the same poses, it will work bad with training with all poses I think
"""Train with PIE test with Kanade. Check the equalization code. """
def crossDataBaseCV():
  # Only train with the frontal pose
  trainData, trainLabels, _, _ = readMultiPieDifferentPoses([2], equalize=args.equalize)
  trainData, trainLabels = shuffle(trainData, trainLabels)

  print "trainLabels"
  print np.argmax(trainLabels, axis=1)

  # for i in xrange(len(trainLabels)):
  #   print "emotions", np.argmax(trainLabels[i])
  #   plt.imshow(vectorToImage(trainData[i], SMALL_SIZE), cmap=plt.cm.gray)
  #   plt.show()
  confustionMatrices = []
  correctAll = []

  params = [(0.001, 0.005), (0.001, 0.05), (0.01, 0.05), (0.01, 0.005)]


  testData, testLabels = readKanade(False, None, equalize=args.equalize, vectorizeLabels=False)
  print "testLabels"
  print testLabels
  # Some emotions do not correspond for a to b, so we have to map them
  testData, testLabels = mapKanadeToPIELabels(testData, testLabels)
  testLabels = labelsToVectors(testLabels, 6)

  print "testLabels after map"
  labelsSimple = np.argmax(testLabels, axis=1)
  print labelsSimple

  # for i in xrange(len(labelsSimple)):
  #   print "emotions", labelsSimple[i]
  #   plt.imshow(vectorToImage(testData[i], SMALL_SIZE), cmap=plt.cm.gray)
  #   plt.show()


  if args.relu:
    activationFunction = Rectified()
    rbmActivationFunctionHidden = RectifiedNoisy()
    rbmActivationFunctionVisible = Identity()
    # unsupervisedLearningRate = 0.05
    # supervisedLearningRate = 0.01
    # momentumMax = 0.95
    trainData = scale(trainData)
    testData = scale(testData)
  else:
    activationFunction = Sigmoid()
    rbmActivationFunctionHidden = Sigmoid()
    rbmActivationFunctionVisible = Sigmoid()
    # unsupervisedLearningRate = 0.05
    # supervisedLearningRate = 0.01
    # momentumMax = 0.95


  for param in params:

    if args.train:
      # TODO: this might require more thought
      net = db.DBN(5, [1200, 1500, 1500, 1500, 6],
                 binary=1-args.relu,
                 activationFunction=activationFunction,
                 rbmActivationFunctionVisible=rbmActivationFunctionVisible,
                 rbmActivationFunctionHidden=rbmActivationFunctionHidden,
                 supervisedLearningRate=param[1],
                 unsupervisedLearningRate=param[0],
                 momentumMax=0.95,
                 nesterovMomentum=args.nesterov,
                 rbmNesterovMomentum=args.rbmnesterov,
                 rmsprop=args.rmsprop,
                 miniBatchSize=args.miniBatchSize,
                 visibleDropout=0.8,
                 hiddenDropout=1.0,
                 rbmHiddenDropout=1.0,
                 rbmVisibleDropout=1.0,
                 preTrainEpochs=args.preTrainEpochs)

      unsupervisedData= buildUnsupervisedDataSetForPIE()

      net.train(trainData, trainLabels, maxEpochs=args.maxEpochs,
                validation=args.validation,
                unsupervisedData=unsupervisedData)

    else:
       # Take the saved network and use that for reconstructions
      with open(args.netFile, "rb") as f:
        net = pickle.load(f)

    probs, predicted = net.classify(testData)

    actualLabels = testLabels
    correct = 0
    errorCases = []

    for i in xrange(len(testLabels)):
      print "predicted"
      print "probs"
      print probs[i]
      print "predicted"
      print predicted[i]
      print "actual"
      actual = actualLabels[i]
      print np.argmax(actual)
      if predicted[i] == np.argmax(actual):
        correct += 1
      else:
        errorCases.append(i)

    print "correct"
    print correct

    print "percentage correct"
    print correct  * 1.0/ len(testLabels)

    print type(predicted)
    print type(actualLabels)
    print predicted.shape
    print actualLabels.shape

    confMatrix = confusion_matrix(np.argmax(actualLabels, axis=1), predicted)

    correctAll += [correct  * 1.0/ len(testLabels)]
    confustionMatrices += [confMatrix]

    print "confusion matrix"
    print confMatrix


  for i, param in enumerate(params):
    print "for param" + str(param)
    print "the correct rate was " + str(correctAll[i])
    print "the confusionMatrix was " + str(confustionMatrices[i])



def addBlobsOfMissingData(testData, sqSize=5):
  maxHeight = SMALL_SIZE[0] - sqSize
  maxLength = SMALL_SIZE[1] - sqSize

  def makeBlob(x):
    x = x.reshape(SMALL_SIZE)
    m = np.random.random_integers(low=0, high=maxHeight)
    n = np.random.random_integers(low=0, high=maxLength)

    for i in xrange(sqSize):
      for j in xrange(sqSize):
        x[m + i, n + j] = 0

    return x.reshape(-1)

  return np.array(map(makeBlob, testData))

def makeMissingDataImage():
  data, labels = readMultiPIE(equalize=args.equalize)
  data, labels = shuffle(data, labels)

  testData = data[0:20]

  testData = addBlobsOfMissingData(testData, sqSize=10)
  final = []
  for i in xrange(6):
    final += [testData[i].reshape(SMALL_SIZE)]

  final = np.hstack(tuple(final))


  plt.imshow(final, cmap=plt.cm.gray, interpolation="nearest")
  plt.axis('off')
  plt.show()

"""Train with PIE test with Kanade. Check the equalization code. """
def missingData():
  data, labels = readMultiPIE(equalize=args.equalize)
  # data, labels = shuffle(data, labels)

  # Random data for training and testing
  kf = cross_validation.KFold(n=len(data), n_folds=5)
  for train, test in kf:
    break

  trainData = data[train]
  trainLabels = labels[train]

  testData = data[test]
  testLabels = labels[test]

  testData = addBlobsOfMissingData(testData, sqSize=10)

  # for i in xrange(10):
  #   plt.imshow(vectorToImage(testData[i], SMALL_SIZE), cmap=plt.cm.gray, interpolation="nearest")
  #   plt.show()

  if args.relu:
    activationFunction = Rectified()
    rbmActivationFunctionHidden = RectifiedNoisy()
    rbmActivationFunctionVisible = Identity()
    unsupervisedLearningRate = 0.005
    supervisedLearningRate = 0.001
    momentumMax = 0.95
    trainData = scale(trainData)
    testData = scale(testData)
  else:
    activationFunction = Sigmoid()
    rbmActivationFunctionHidden = Sigmoid()
    rbmActivationFunctionVisible = Sigmoid()
    unsupervisedLearningRate = 0.05
    supervisedLearningRate = 0.01
    momentumMax = 0.95

  if args.train:
    # TODO: this might require more thought
    net = db.DBN(5, [1200, 1500, 1500, 1500, 6],
               binary=1-args.relu,
               activationFunction=activationFunction,
               rbmActivationFunctionVisible=rbmActivationFunctionVisible,
               rbmActivationFunctionHidden=rbmActivationFunctionHidden,
               unsupervisedLearningRate=unsupervisedLearningRate,
               supervisedLearningRate=supervisedLearningRate,
               momentumMax=momentumMax,
               nesterovMomentum=args.nesterov,
               rbmNesterovMomentum=args.rbmnesterov,
               rmsprop=args.rmsprop,
               miniBatchSize=args.miniBatchSize,
               visibleDropout=0.8,
               hiddenDropout=1.0,
               rbmHiddenDropout=1.0,
               rbmVisibleDropout=1.0,
               preTrainEpochs=args.preTrainEpochs)

    unsupervisedData = buildUnsupervisedDataSetForPIE()

    net.train(trainData, trainLabels, maxEpochs=args.maxEpochs,
              validation=args.validation,
              unsupervisedData=unsupervisedData,
              trainingIndices=train)

    if args.save:
      with open(args.netFile, "wb") as f:
        pickle.dump(net, f)

  else:
     # Take the saved network and use that for reconstructions

    print "using ", args.netFile, " for reading the pickled net"
    with open(args.netFile, "rb") as f:
      net = pickle.load(f)

      trainingIndices = net.trainingIndices
      testIndices = np.setdiff1d(np.arange(len(data)), trainingIndices)
      testData = data[testIndices]
      print "len(testData)"
      print len(testData)
      testData = addBlobsOfMissingData(testData, sqSize=5)


  print net.__dict__

  probs, predicted = net.classify(testData)

  actualLabels = testLabels
  correct = 0
  errorCases = []

  for i in xrange(len(testLabels)):
    print "predicted"
    print "probs"
    print probs[i]
    print "predicted"
    print predicted[i]
    print "actual"
    actual = actualLabels[i]
    print np.argmax(actual)
    if predicted[i] == np.argmax(actual):
      correct += 1
    else:
      errorCases.append(i)

  print "correct"
  print correct

  print "percentage correct"
  print correct  * 1.0/ len(testLabels)

  print type(predicted)
  print type(actualLabels)
  print predicted.shape
  print actualLabels.shape

  confMatrix = confusion_matrix(np.argmax(actualLabels, axis=1), predicted)

  print "confusion matrix"
  print confMatrix

def makeMissingDataOnly12Positions(testData):
  def makeBlob(x):
    x = x.reshape(SMALL_SIZE)
    m = np.random.randint(low=0, high=4)
    n = np.random.randint(low=0, high=3)

    for i in xrange(10):
      for j in xrange(10):
        x[10 * m  + i, 10 * n + j] = 0

    return x.reshape(-1), (m,n)

  data = []
  coordinates = []
  for i, d in enumerate(testData):
    d, (m, n) = makeBlob(d)
    data += [d]
    coordinates += [(m,n)]

  return np.array(data), coordinates

def missingDataTestFromTrainedNet():
  data, labels = readMultiPIE(equalize=args.equalize)
  data, labels = shuffle(data,labels)


  # with open(args.netFile, "rb") as f:
  #   net = pickle.load(f)

  # trainingIndices = net.trainingIndices
  # testIndices = np.setdiff1d(np.arange(len(data)), trainingIndices)

  # print testIndices
  # testData = data[testIndices]
  # testLabels = labels[testIndices]
  # print "len(testData)"
  # print len(testData)
   # Random data for training and testing
  kf = cross_validation.KFold(n=len(data), n_folds=5)
  for train, test in kf:
    break

  trainData = data[train]
  trainLabels = labels[train]

  testData = data[test]
  testLabels = labels[test]

  testData, pairs = makeMissingDataOnly12Positions(testData)


  # testData = addBlobsOfMissingData(testData, sqSize=10)

  # for i in xrange(10):
  #   plt.imshow(vectorToImage(testData[i], SMALL_SIZE), cmap=plt.cm.gray, interpolation="nearest")
  #   plt.show()

  if args.relu:
    activationFunction = Rectified()
    rbmActivationFunctionHidden = RectifiedNoisy()
    rbmActivationFunctionVisible = Identity()
    unsupervisedLearningRate = 0.005
    supervisedLearningRate = 0.001
    momentumMax = 0.95
    trainData = scale(trainData)
    testData = scale(testData)
  else:
    activationFunction = Sigmoid()
    rbmActivationFunctionHidden = Sigmoid()
    rbmActivationFunctionVisible = Sigmoid()
    unsupervisedLearningRate = 0.05
    supervisedLearningRate = 0.01
    momentumMax = 0.95

  if args.train:
    # TODO: this might require more thought
    net = db.DBN(5, [1200, 1500, 1500, 1500, 6],
               binary=1-args.relu,
               activationFunction=activationFunction,
               rbmActivationFunctionVisible=rbmActivationFunctionVisible,
               rbmActivationFunctionHidden=rbmActivationFunctionHidden,
               unsupervisedLearningRate=unsupervisedLearningRate,
               supervisedLearningRate=supervisedLearningRate,
               momentumMax=momentumMax,
               nesterovMomentum=args.nesterov,
               rbmNesterovMomentum=args.rbmnesterov,
               rmsprop=args.rmsprop,
               miniBatchSize=args.miniBatchSize,
               visibleDropout=0.8,
               hiddenDropout=1.0,
               rbmHiddenDropout=1.0,
               rbmVisibleDropout=1.0,
               preTrainEpochs=args.preTrainEpochs)

    unsupervisedData = buildUnsupervisedDataSetForPIE()

    net.train(trainData, trainLabels, maxEpochs=args.maxEpochs,
              validation=args.validation,
              unsupervisedData=unsupervisedData,
              trainingIndices=train)

    if args.save:
      with open(args.netFile, "wb") as f:
        pickle.dump(net, f)

  dictSquares = {}
  for i in xrange(4):
    for j in xrange(3):
      dictSquares[(i,j)] = []

  # for i in xrange(10):
  #   plt.imshow(vectorToImage(testData[i], SMALL_SIZE), cmap=plt.cm.gray, interpolation="nearest")
  #   plt.show()

  probs, predicted = net.classify(testData)

  actualLabels = testLabels
  correct = 0
  errorCases = []

  for i in xrange(len(testLabels)):
    print "predicted"
    print "probs"
    print probs[i]
    print "predicted"
    print predicted[i]
    print "actual"
    actual = actualLabels[i]
    print np.argmax(actual)
    if predicted[i] == np.argmax(actual):
      correct += 1
      dictSquares[pairs[i]] += [1]
    else:
      errorCases.append(i)
      dictSquares[pairs[i]] += [0]

  print "percentage correct"
  print correct  * 1.0/ len(testLabels)

  mat = np.zeros((4, 3))
  for i in xrange(4):
    for j in xrange(3):
      print "len(dictSquares[(i,j)])"
      print len(dictSquares[(i,j)])
      mat[i,j] = sum(dictSquares[(i,j)]) * 1.0 / len(dictSquares[(i,j)])


  print mat
  plt.matshow(mat, cmap=plt.get_cmap("YlOrRd"),interpolation='none')
  plt.show()


def main():
  if args.rbm:
    rbmEmotions()
  if args.cv:
    deepbeliefKanadeCV()
  if args.dbKanade:
    deepbeliefKanade()
  if args.dbPIE:
    deepbeliefMultiPIE()
  if args.cvPIE:
    deepbeliefPIECV()
  if args.svmPIE:
    svmPIE()
  if args.crossdb:
    crossDataBase()
  if args.crossdbCV:
    crossDataBaseCV()

  if args.illumination or args.pose or args.subjects:
    deepBeliefPieDifferentConditions()

  if args.missing:
    missingData()

  if args.average:
    deepbeliefMultiPIEAverage()

  if args.kaggle:
    deepbeliefKaggleCompetition()


# You can also group the emotions into positive and negative to see
# if you can get better results (probably yes)
if __name__ == '__main__':
  # import random
  # print "FIXING RANDOMNESS"
  # random.seed(6)
  # np.random.seed(6)
  # missingDataTestFromTrainedNet()
  main()