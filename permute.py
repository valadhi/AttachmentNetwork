import itertools
import numpy as np 
def frange(x, y, jump):
  while x <= y:
    yield round(x,1)
    x += jump

securestart = [(0.7,1.0), (0.0,0.2), (0.0,0.2)]
avoidantstart = [(0.0,0.3), (0.4,0.6), (0.4,0.6)]
ambivalentstart = [(0.2,0.4), (0.2,0.4), (0.2,0.4)]
startlist = [securestart, avoidantstart, ambivalentstart]
securelist = []
avoidlist = []
ambivlist = []
outlist = [securelist, avoidlist, ambivlist]

for start in xrange(len(startlist)):
	foo = []
	print start
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

for i in outlist:
	print np.array(i)
