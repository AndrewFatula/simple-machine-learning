import time
import os
import math
import copy
import numpy as np
import pandas as pd
import random
import sklearn.metrics as sk_m
import matplotlib.pyplot as plot 
from scipy.stats import mstats as sc_st_mst
from sklearn.tree import DecisionTreeRegressor

start = time.localtime(time.time())


#Generates our test and training data with multivariate normal distribution
random.seed(10)
n = 4000
x = [np.random.multivariate_normal([3.5*i-1,i,2.5*i,0.5*i,1.1*i+6,2.2*i-2],np.array([[1,0.5,0.3,0.4,0.3,0.5],
														  		   					 [0.5,1,0.3,0.4,0.5,0.3],
														  		    				 [0.3,0.3,1,0.5,0.3,0.5],
														  		   					 [0.4,0.4,0.5,1,0.7,0.3],
																   					 [0.3,0.5,0.3,0.7,1,0.2],
														  		   					 [0.5,0.3,0.5,0.3,0.2,1]]),n) for i in range(1,6)]
x = np.vstack(x).astype(np.float64)


columns = ['col'+str(i) for i in range(1,7)]


data=pd.DataFrame(x, columns = columns)
y_variable = columns[-1]
variables = list(data.columns)
variables.remove(y_variable)
print variables

#dividing our existed data on training and test parts
def train_test_split(data, size):
	data['split'] = np.random.rand(len(data.index))
	train = data[data['split']>size]
	train = train.drop('split',1)
	test = data[data['split']<=size]
	test = test.drop('split',1)
	return train, test

train, test = train_test_split(data, 0.1) 	

#function that finds split for each node
def get_split(data, variables, y_variable, min_samples_leaf, n_quantiles):
	
	variance = np.var(data[y_variable])
	split_value = None
	#for each feature
	for variable in variables:
		value_list = data[variable]
		if len(np.unique(value_list))>n_quantiles:
			#set the quantiles
			probs = [j/float(n_quantiles) for j in range(1,n_quantiles+1)]
			values = sc_st_mst.mquantiles(value_list,probs)

		else:
			if len(np.unique(value_list))==1:
				continue
			values = np.unique(value_list)	
		#for each value of given feature		
		for value in values[:-1]:

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

#Node class that saves all needed for as parameters of Decision tree
class Node():
	def __init__(self, parent, length, is_right):

		self.variable = None		
		self.value = None
		self.parent = parent
		self.variance = None
		self.is_leaf = False
		self.height = None
		self.y_value = None
		self.left_child = None
		self.right_child = None
		self.length = length
		self.is_right = is_right
		self.root_node = None
		
	
		
		
#Function that recursively builds the tree	
def compute_tree(data, variables, y_variable, max_height, min_samples_split = 1, 
				min_samples_leaf = 1, n_quantiles=10, parent=None, length = None, is_right = False):
	
	
	node = Node(parent, length, is_right)
	#For each node it checks hyper parameters that were set
	if node.parent == None:
		node.root_node = True
		node.height = 0
		node.variance = np.var(data[y_variable])
	else:
		node.height = node.parent.height + 1
	
	if (node.length != None) and (node.length < min_samples_split):
		print 'Node_split ; complex_node'
		node.is_leaf = True
		node.y_value = np.mean(data[y_variable])
		return node
															
	if node.variance == 0:
		print 'Impossible zero variance'
		node.is_leaf = True
		node.y_value = np.mean(data[y_variable])
		return node	

	if node.height == max_height:
		print 'Height ; complex_node'
		node.is_leaf = True
		node.y_value = np.mean(data[y_variable])
		return node
		
	#for each node it finds the best split and saves parameters of that split
	parameters = get_split(data, variables, y_variable, min_samples_leaf, n_quantiles)
	print parameters[2]
	print len(data)

	if parameters == None:
		node.is_leaf= True
		node.y_value = np.mean(data[y_variable])
		return node

	node.variable = parameters[0]
	node.value = parameters[1]
	node.variance = parameters[2]

		
	#for each node dividing our data on twoo splitted parts
	data_for_right_branch = data[data[node.variable] <= node.value]
	data_for_left_branch = data[data[node.variable] > node.value]
	right = len(data_for_right_branch.index) 
	left = len(data_for_left_branch.index)
		


	#and for each splitted part repeat the procces
	node.right_child = compute_tree(data_for_right_branch, variables, y_variable, max_height,
										min_samples_split = min_samples_split, n_quantiles = n_quantiles, parent = node, length = right, is_right = True)	
	node.left_child = compute_tree(data_for_left_branch, variables, y_variable, max_height,
										min_samples_split = min_samples_split, n_quantiles = n_quantiles, parent = node, length = left)
	


	return node

tree = compute_tree(data, variables, y_variable, 10, min_samples_split = 1000, min_samples_leaf = 1000, n_quantiles = 100)


	
sk_tree = DecisionTreeRegressor(max_depth = 10)
sk_tree.fit(train[variables].values, train[y_variable].values)


###Functions for evaluating the tree
def mean_error(actual, predicted):
	return sum(abs(actual-predicted)), np.mean(abs(actual-predicted))

def square_error(actual, predicted):
	return sum((actual-predicted)**2)	

def r_squared(actual, predicted):
	mean_y = np.mean(actual)
	res = sum((predicted - actual)**2)
	tot = sum((actual - mean_y)**2)
	return 1 - res/float(tot)	
###

#function that counts number of node of all branches of builded tree
def count_nodes(node,i=0):
	i+=1
	if (node.is_leaf) :
		return i
	return count_nodes(node.left_child,i) + count_nodes(node.right_child,i)


#function that counts leaf nodes of all branches of builded tree
def count_leaves(node):
	if (node.is_leaf) :
		return 1
	return count_leaves(node.left_child) + count_leaves(node.right_child)
		 	
#function that for each row sets the score with our builded tree
def score(row, node, variables):
	if node.is_leaf:
		return	node.y_value	
	if row[variables.index(node.variable)] <= node.value:
		return score(row,node.right_child, variables)
	else:
		return score(row,node.left_child, variables)
#function that makes predictions with our builded tree
def predictions(data, node, variables, class_variable):
	length = len(data.index)
	
	data_arr = np.array(data)
	
	return [score(data_arr[i], node, variables) for i in range(length)] 
		


test_data = test.copy()		
predicted = predictions(test, tree, variables, y_variable)



tested_arr = np.array(predicted)
test_arr = np.array(test_data[y_variable])
sk_predicted = sk_tree.predict(test[variables].values)

print mean_error(test_arr, tested_arr) , '  --  Mean error of predictions'

print r_squared(test_arr, tested_arr) , '  --  R_squared'

print mean_error(test_arr, sk_predicted) , '  --  Mean_error sklearn'

print r_squared(test_arr, sk_predicted) , '  --  R_squared sklearn'
			

print 'leaves : %s' % (count_leaves(tree))
print 'nodes : %s' % (count_nodes(tree))


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


