import time
import os
import math
import copy
import numpy as np
import pandas as pd
import random
import sklearn.metrics as sk_m
import matplotlib.pyplot as plot 
import numpy.linalg as np_lin



random.seed(10)
n = 5000
x1 = np.random.multivariate_normal([0,1.2,1],[[0.9,0.6,0.5],[0.6,0.9,0.75],[0.5,0.75,0.9]],n)
x2 = np.random.multivariate_normal([0.8,1,1.7],[[0.9,0.6,0.5],[0.6,0.9,0.75],[0.5,0.75,0.9]],n)
x = np.vstack((x1,x2)).astype(np.float64)
y = np.hstack((np.zeros(n),np.ones(n)))

plot.figure(figsize=(10,7))
plot.scatter(x[:,0],x[:,1],s = 7, c = np.hstack((['red']*n,['blue']*n)), alpha = 0.2)
plot.show()


start = time.localtime(time.time())

def sigmoid(scores):
	return 1 / (1 + np.exp(-scores))

def log_likehood(x,y,weights):
	scores = np.dot(x,weights)
	log_lhood = np.sum(y*scores - np.log(1+np.exp(scores)))
	return log_lhood 

def predict(predictions):
	pred = []
	for y in predictions:
		if y > 0.5:
			pred.append(1)
		else:
			pred.append(0)

	return np.array(pred)	

def sigmoid(scores):
	return 1 / (1 + np.exp(-scores))

def logistic_regression(x, y, n, alpha):
	weights = np.zeros(x.shape[1])
	
	accuracy = 0
	auc = 0
	result = None
	for step in range(n):
		scores = np.dot(x, weights)
		predictions = sigmoid(scores)
		errors = (predictions - y)
		H_multiplier = np.diag(predictions*(1-predictions))
		hessian = np.matrix(np.dot(np.dot(x.T,H_multiplier),x))
		gradient = np.dot(x.T, errors)
		weights = np.ravel(weights - np.dot(hessian.I,gradient)*alpha)
	
		del errors
		del predictions
		del scores
		del H_multiplier
		if step % 10 == 0:
			alpha = alpha*3
		if step % 4 == 0:
			possibilities = np.dot(x, weights)
			predictions = predict(possibilities)
			if (accuracy >= sk_m.accuracy_score(y, predictions)) and 
						(auc >= sk_m.roc_auc_score(y, predictions)):
				return result
			accuracy = sk_m.accuracy_score(y, predictions)
			auc = sk_m.roc_auc_score(y, predictions)
			result = weights

	return weights		

weights = logistic_regression(x, y, 40, 0.05)

scores = np.dot(x,weights)

possibilities = sigmoid(scores)

predictions = predict(possibilities)

print sk_m.accuracy_score(y, predictions)
print sk_m.roc_auc_score(y, predictions)
print sk_m.precision_score(y, predictions)
print sk_m.confusion_matrix(y, predictions)



end = time.localtime(time.time())

start_in_sec = start[3]*3600 + start[4]*60 + start[5]
end_in_sec = end[3]*3600 + end[4]*60 + end[5]


all_time_min = int((end_in_sec-start_in_sec)/60)
all_time_sec = (end_in_sec-start_in_sec)%60
if all_time_min < 10:
	if all_time_sec < 10:
		print('0%s:0%s' % (all_time_min, all_time_sec ))
	else:
		print('0%s:%s' % (all_time_min, all_time_sec ))	
else:
	if all_time_sec < 10:
		print('%s:0%s' % (all_time_min, all_time_sec ))
	else:
		print('%s:%s' % (all_time_min, all_time_sec ))


