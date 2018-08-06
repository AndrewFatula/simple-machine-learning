
import time
import os
import math
import copy
import numpy as np
import pandas as pd
from collections import Counter
import random
import sklearn.metrics as sk_m
from scipy.stats import mstats as sc_st_mst
from matplotlib import pyplot as plt
from sklearn import datasets

start = time.localtime(time.time())

new_digits = datasets.load_digits()


images = np.array(new_digits['images'])


test_indices = np.random.randint(0,len(images),int(len(images)/10))
train_indices = list(set(np.arange(len(images)))-set(test_indices))

train_images = new_digits['images'][train_indices]
train_labels = np.array(new_digits['target'])[train_indices]

test_images = new_digits['images'][test_indices]
test_labels = np.array(new_digits['target'])[test_indices]

def image_converter(images):
	return np.array([np.ravel(image) for image in images])

train_images = image_converter(train_images)	
test_images = image_converter(test_images)

class KNN_classifier:

	def __init__(self, n):
		self.features = None
		self.labels = None
		self.number_of_neighbors = n


	def fit(self, x, y):
		self.labels = y
		self.features = x
		self.classes = np.unique(self.labels)

	def predict(self, test_x):

		if len(np.shape(test_x))==1:

			differences = sum(abs(self.features-test_x).T)

			min_indices = differences.argsort()[:self.number_of_neighbors]

			key_labels = self.labels[min_indices]

			label = Counter(key_labels).most_common()[0][0]

			return label

		else:	
			print('ERROR.predictions should be made iteratively')
		
		



predictor = KNN_classifier(15)
predictor.fit(train_images, train_labels)
predicted = [predictor.predict(test_sample) for test_sample in test_images]
actual = copy.copy(test_labels)


print(sum(np.where(actual==predicted,1,0))/len(actual))

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


				


