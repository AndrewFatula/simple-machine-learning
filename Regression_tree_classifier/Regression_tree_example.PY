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
from scipy.stats import mstats as sc_st_mst
from sklearn.linear_model import LinearRegression
import datetime

start = time.localtime(time.time())

os.chdir('data')
#Data of air quality
data = pd.read_excel('AirQualityUCI.xlsx', na_values = '?')
print len(data.index)
temp = 'T'
Time = data.select_dtypes(include = 'object')
date = data.select_dtypes(include = 'datetime64').astype(object)
colnames = list(data.columns)
colnames.remove('Date')
colnames.remove('Time')	
colnames.remove(temp)



#compute correlation beeween a nd b
def correlation(a,b):
	sum_ab = sum(a*b)
	m_ab = sum_ab/float(len(a))
	return (m_ab - np.mean(a)*np.mean(b))/float(np.std(a)*np.std(b))

#converts all not numeric features to numeric
def Date_converter(data):
	time = []
	for t in data['Time']:
		for i in range(0,24):
			if i == 23:
				if(datetime.time(i,0,0) <= t):
					time.append(i)
			else:
				if(datetime.time(i,0,0) <= t) and (datetime.time(i+1,0,0) > t ):
					time.append(i)		
	data['time'] = time			
	data2004 = data[data['Date'] < np.datetime64('2005-01-01')] 
	data2005 = data[data['Date'] >= np.datetime64('2005-01-01')]
	val2004 = np.unique(data2004['Date'])
	val2005 = np.unique(data2005['Date'])
	data['date'] = range(len(data.index))
	for i in range(len(val2004)):
		data['date'][data['Date']==val2004[i]]=i
	for j in range(len(val2005)):
		data['date'][data['Date']==val2005[j]]=j
	columns = ['time','date']
	return columns	

timecol = Date_converter(data) 
			
	
#split the data on training and test set
def train_test_split(data, size):
	data['split'] = np.random.rand(len(data.index))
	train =  data[data['split']>size]
	test =  data[data['split']<=size]
	data = data.drop('split',1)
	train = train.drop('split',1)
	test = test.drop('split',1)
	return train, test

#computes mean error of predictions
def mean_error(actual, predicted):
	return sum(abs(actual-predicted)), np.mean(abs(actual-predicted))
#computes R_squared of predictions
def r_squared(actual, predicted):
	mean_y = np.mean(actual)
	res = sum((predicted - actual)**2)
	tot = sum((actual - mean_y)**2)
	return 1 - res/float(tot)	

#finds the best split of given data
def get_split(data, variables, y_variable, min_samples_leaf, n_quantiles):
	
	variance = np.var(data[y_variable])
	split_value = None
	#for each feature in the given data
	for variable in variables:
		value_list = data[variable]
		if len(np.unique(value_list))>n_quantiles:
			#finding the quantiles of given features
			probs = [j/float(n_quantiles) for j in range(1,n_quantiles+1)]
			values = sc_st_mst.mquantiles(value_list,probs)

		else:
			if len(np.unique(value_list))==1:
				continue
			values = np.unique(value_list)	
				
		for value in values[:-1]:
			#for each unique value of given feature
			data_with_value = data[data[variable] <= value]
			data_without_value =  data[data[variable] > value]
			without_len = len(data_without_value.index)
			with_len = len(data_with_value.index)
			if (with_len < min_samples_leaf) or (without_len < min_samples_leaf):
				continue	
		
			### Ratios of each value of specified variable
			ratio = with_len/float(len(data.index))

			### split_entropy shows how good split seperates class_values in generaly 
			
			split_variance =  ratio*np.var(data_with_value[y_variable])+(1-ratio)*np.var(data_without_value[y_variable])

				
			if split_variance < variance :
				variance = split_variance
				split_variable = variable
				split_value = value

	if split_value == None:
		return None		
				
	

	return  split_variable, split_value, variance

####
class Node():
	def __init__(self, parent, length, is_right):
		
		#for each split must be defined :
		#feature, where we found value for the best split
		#value which we split on of that feature
		#previous node, which is the parent node
		#variance indicator of given data
		#bool that gives as information if it is a leaf node
		#heigth of that node (sequential number of given split that is saved of current node)
		#if it is a leaf node, class value of that leaf node
		#nodes for which current node is a parent node
		#number of rows of training data that is produced by previous split
		#the bool that gives information if current node is on the rigth branch of the previous split
		#the bool that gives us information if given node is the root node
		self.variable = None		
		self.value = None
		self.parent = parent
		self.variance = None
		self.is_leaf = False
		self.height = None
		self.model = None
		self.left_child = None
		self.right_child = None
		self.length = length
		self.is_right = is_right
		self.root_node = None
		
	
		
		
#Function that builds the tree recutrsively
def compute_tree(data, colnames, variables, y_variable, max_height, min_samples_split = 1, 
				min_samples_leaf = 1, n_quantiles=10, parent=None, length = None, is_right = False):
	#for each node of the tree
	node = Node(parent, length, is_right)
	#check if node has reached hyper_parameters, that were set when the funcion was called
	if node.parent == None:
		node.root_node = True
		node.height = 0
		node.variance = np.var(data[y_variable])
	else:
		node.height = node.parent.height + 1
	
	#if node is reaching the hyper_parameters: for the data splitted by previous node is using linear_regression model from the sklearn framework
	if (node.length != None) and (node.length < min_samples_split):
		node.is_leaf = True
		node.model = LinearRegression()
		node.model.fit(np.array(data[colnames]), np.array(data[temp]))
		return node
															
	if node.variance == 0:
		node.is_leaf = True
		node.model = LinearRegression()
		node.model.fit(np.array(data[colnames]), np.array(data[temp]))
		return node	

	if node.height == max_height:
		node.is_leaf = True
		node.model = LinearRegression()
		node.model.fit(np.array(data[colnames]), np.array(data[temp]))
		return node
	#finding the split and saving the split parameters
	parameters = get_split(data, variables, y_variable, min_samples_leaf, n_quantiles)

	if parameters == None:
		node.is_leaf= True
		node.model = LinearRegression()
		node.model.fit(np.array(data[colnames]), np.array(data[temp]))
		return node

	node.variable = parameters[0]
	node.value = parameters[1]
	node.variance = parameters[2]
	#dividing node data for the next use of get_split function by the given split parameters
	data_for_right_branch = data[data[node.variable] <= node.value]
	data_for_left_branch = data[data[node.variable] > node.value]
	right = len(data_for_right_branch.index) 
	left = len(data_for_left_branch.index)
	#repeating the procces recursively for each splitted part 
	node.right_child = compute_tree(data_for_right_branch, colnames, variables, y_variable, max_height,
										min_samples_split = min_samples_split, min_samples_leaf = min_samples_leaf, n_quantiles = n_quantiles, parent = node, length = right, is_right = True)	
	node.left_child = compute_tree(data_for_left_branch, colnames,variables, y_variable, max_height,
										min_samples_split = min_samples_split, min_samples_leaf = min_samples_leaf, n_quantiles = n_quantiles, parent = node, length = left)

	return node	

#for each row of test data compute the score with the builded decision tree
def score(row, node, variables):
	if node.is_leaf:
		return	node.model.predict(np.array(row[variables]).reshape(1, -1)).item(0)	
	if row[node.variable] <= node.value:
		return score(row,node.right_child, variables)
	else:
		return score(row,node.left_child, variables)


#making predictions for test data	
def predictions(data, node, variables):
	length = len(data.index)
	return [score(data.iloc[i], node, variables) for i in range(length)]		


#function that finds the best hyper_parameters of the decision tree for the given data
def cross_validation(data, timecol, colnames, temp, ratio,):
	best_nodes, best_n_samples = None, None
	best_result = len(data) 
	for i in range(5,16):
		for j in range(1,11):
			result = 0
			for i in range(5):
				train, test= train_test_split(data, ratio)
				tree = compute_tree(train, colnames, timecol, temp, i, min_samples_split = j*100, min_samples_leaf = j*100, n_quantiles = 100)	
				predicted = np.array(predictions(test, tree, colnames))
				actual = np.array(test[temp])
				result += mean_error(actual, predicted)[1]
			print i,j,result
			if result < best_result:
				best_result = result
				best_nodes = i
				best_n_samples = j
	return best_nodes, best_n_samples

	


			 



#counts the number of  nodes on all branches of the tree
def count_nodes(node,i=0):
	i+=1
	if (node.is_leaf) :
		return i
	return count_nodes(node.left_child,i) + count_nodes(node.right_child,i)


#counts the number of leaf nodes on all branches of the tree
def count_leaves(node):
	if (node.is_leaf) :
		return 1
	return count_leaves(node.left_child) + count_leaves(node.right_child)
		 	



				





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




for i in range(1,10):
	for j in range(17):
		train, test = train_test_split(data,0.1)

		setting_score_tree = compute_tree(train, colnames, timecol, temp, j+3, min_samples_split = i*100, min_samples_leaf = i*100, n_quantiles = 100)

		predicted = np.array(predictions(test, setting_score_tree, colnames))
		actual = np.array(test[temp])

		print '\n\n mean_error of predictions : ' , mean_error(actual, predicted)[1]	
		print'\n r_squared of predictions : ', r_squared(actual,predicted)
		print '\n Hyper parameters: minSamplesLeaf = %s, minSamplesSplit = %s, maxHeight = %s, nQuantiles = %s' % (i*100,i*100,j+3,100)


