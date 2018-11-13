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


start = time.localtime(time.time())


#Data class with methods, needed for preparing the dta for classification tasks
class my_data():
	def __init__(self):
		self.all_examples = None
		self.defined = None
		self.undefined = None
		self.variables = []
		self.class_variable = None
		self.classes = []
		self.p_initial = None
		self.defined_test = None
		self.defined_train = None
			
	#Read_data method reads the data and save all needed in classification data properties

	def read_data(self, directory, file):
		os.chdir(directory)

		self.all_examples = pd.read_csv(file, sep = ',', header = 0, dtype = 'float', na_values = '?')
		variables = list(data.all_examples.columns)
 		
 		#saves all features names
		self.variables = copy.copy(variables)
		self.variables.remove(variables[-1])
		#saves label name
		self.class_variable = variables[-1]
		#saves all unique label classes
		classes = np.unique(data.all_examples[data.class_variable])
		self.classes = list(classes[0:3])
		#separates data with defined and undefined labels
		self.defined = data.all_examples[data.all_examples[data.class_variable].notna()]
		self.undefined = data.all_examples[data.all_examples[data.class_variable].isna()]
		#saves initial ration of twoo classes in label
		self.p_initial = (len(self.all_examples[self.all_examples[self.class_variable]==self.classes[0]].index)/
			float(len(self.all_examples.index)))

	#Method that for each N/A value in features puts in compliance the possibility of positive label if value of selected feature

	def fill_missing_values(self):
		for variable in self.variables:
			if 	len(self.defined[self.defined[variable].isna()].index) > 0:
				p_na = (len(self.defined[(self.defined[variable].isna()) & (self.defined[self.class_variable]==self.classes[1])].index)/
					float(len(self.defined[self.defined[variable].isna()].index)))
				self.defined[variable] = self.defined[variable].fillna(p_na*np.mean(self.defined[variable]))		

	#Method for deviding data on test and training set

	def train_test_split(self, size):
		self.defined['split'] = np.random.rand(len(self.defined.index))
		self.defined_train = self.defined[self.defined['split']>size]
		self.defined_test = self.defined[self.defined['split']<=size]
		self.defined_train = self.defined_train.drop('split',1)
		self.defined_test = self.defined_test.drop('split',1)


data = my_data()
data.read_data('\Python27\Binary_classifiers\my\data','Data.csv')
data.fill_missing_values()
data.train_test_split(0.2)


#Function that returns indicator of information entropy considering initial distribution of label classes

def get_entropy(p, p_initial):
	if p <= p_initial:
		p_current = p/(2*p_initial)
		entropy = -(p_current*math.log(p_current+0.0001,2)+(1-p_current)*math.log(1-p_current+0.0001,2))
		
	else:
		p_current = (p+1-2*p_initial)/(2*(1-p_initial))
		entropy = -(p_current*math.log(p_current+0.0001,2)+(1-p_current)*math.log(1-p_current+0.0001,2))
	return entropy 


#Method that makes a split for eack node in Decision tree
#it returns split feature, split value of that feature, entropy indicator of found split and ratios of classes in label of each splitted part of the data

def get_split(data, variables, class_variable, classes, p_initial, min_samples_leaf):

	entropy = 1
	split_variable = None
	split_value = None
	ones_r = None
	ones_l = None

	#for each feature in dataset
	for variable in variables:
		value_list = data[variable]

		if len(np.unique(value_list))>20:
			
			#finding 20 quantiles for dividing our data on given feature in its quantile

			probs = [j/20.0 for j in range(1,21)]
			values = sc_st_mst.mquantiles(value_list,probs)

		else:

			if len(np.unique(value_list))==1:
				continue
			values = np.unique(value_list)	
		
		#for each quantile of the given feature
		for value in values[:-1]:

			data_with_value = data[data[variable] <= value]
			data_without_value =  data[data[variable] > value]
			without_len = len(data_without_value.index)
			with_len = len(data_with_value.index)
			if (with_len < min_samples_leaf) or (without_len < min_samples_leaf):
				continue	
	
			### Ratios of each value of specified variable
			p_value = with_len/float(len(data.index))

			ones_with = len(data_with_value[data_with_value[class_variable]==classes[0]].index)
			ones_without = len(data_without_value[data_without_value[class_variable]==classes[0]].index)

			p_with = ones_with/float(with_len)
			p_without = ones_without/float(without_len)
			### split_entropy shows how good split seperates label classes in generaly 
			split_entropy = p_value*get_entropy(p_with, p_initial) + (1-p_value)*get_entropy(p_without, p_initial)
			
			
			if split_entropy < entropy :
				p_right = p_with
				p_left = p_without
				entropy = split_entropy
				split_variable = variable
				split_value = value

	if split_variable == None:
		return None		
				
	

	return split_variable, split_value, entropy, p_right, p_left

####
#Node clas that saves all needed iformation about the best splits for building the decision tree
class Node():
	def __init__(self, parent, length, is_right):
		#for each split must be defined :
		#feature, where we found value for the best split
		#value which we split on of that feature
		#previous node, which is the parent node
		#entropy indicator of gaven split 
		#bool that gives as information if it is a leaf node
		#heigth of that node (sequential number of given split that is saved of current node)
		#if it is a leaf node, class value of that leaf node
		#nodes for which current node is a parent node
		#number of rows of training data that is produced by previous split
		#the bool that gives information if current node is on the rigth branch of the previous split
		self.variable = None
		self.value = None
		self.parent = parent
		self.entropy = None
		self.is_leaf = False
		self.height = None
		self.class_value = None
		self.left_child = None
		self.right_child = None
		self.length = length
		self.is_right = is_right
		
	
		
		
# function that recursively builds the decision tree, which is called for each node of our tree		
def compute_tree(data, variables, class_variable, classes, p_initial, max_height,
				 min_samples_split = 1, min_samples_leaf = 1, parent=None, length = None,
				  p_current = None, is_right = False):
	#defining node for which we want to find the best split or to make it a leaf node
	node = Node(parent, length, is_right)

	# This function is checking for each node if it is reached the Hyper_parameters of the tree, if they are not reached functions finds the best split for that node

	if node.parent == None:
		node.height = 0
		p_current = p_initial
	else:
		node.height = node.parent.height + 1
	
	if (node.length != None) and (node.length < min_samples_split):
		print 'Node_split ; complex_node'
		node.is_leaf = True
		if p_current > 0.5:
			node.class_value = classes[0]
		else:
			node.class_value = classes[1]	
		return node
																							

	if p_current == 1:
		print 'Zeros'
		node.is_leaf = True
		node.class_value = 0
		return node

	elif p_current == 0:
		print 'Ones'
		node.is_leaf = True
		node.class_value = 1
		return node	

	if node.height == max_height:
		print 'Height ; complex_node'
		node.is_leaf = True
		if p_current > 0.5:
			node.class_value = classes[0]
		else:
			node.class_value = classes[1]
		return node
		
	#for each node we need to fing and save parameters of its split
	parameters = get_split(data, variables, class_variable, classes, p_initial, min_samples_leaf)

	if parameters == None:
		node.is_leaf= True
		if p_current > 0.5:
			node.class_value = classes[0]
		else:
			node.class_value = classes[1]
		return node

	node.variable = parameters[0]
	node.value = parameters[1]
	node.entropy = parameters[2]

	
	entropy = get_entropy(p_current, p_initial)

	if node.entropy-0.001 > entropy:
		print 'Entropy ; complex_node'
		node.is_leaf = True
		if p_current > 0.5:
			node.class_value = classes[0]
		else:
			node.class_value = classes[1]
		return node
		
	# splitting our training data with our split faeture and its value in order to build out tree
	data_for_right_branch = data[data[node.variable] <= node.value]
	data_for_left_branch = data[data[node.variable] > node.value]
	right = len(data_for_right_branch.index) 
	left = len(data_for_left_branch.index)
		
	#for each part of our splitted data we use our computing tree function recursively
	node.right_child = compute_tree(data_for_right_branch, variables, class_variable, classes, p_initial, max_height,
										min_samples_split = min_samples_split, parent = node, length = right, p_current = parameters[3], is_right = True)	
	node.left_child = compute_tree(data_for_left_branch, variables, class_variable, classes, p_initial, max_height,
										min_samples_split = min_samples_split, parent = node, length = left, p_current = parameters[4])
	


	return node





tree = compute_tree(data.defined_train, data.variables, data.class_variable, data.classes, data.p_initial, 20, min_samples_split = 100)


#function for counting how much node our tree has on all of its branches
def count_nodes(node,i):
	i+=1
	if node.is_leaf:
		return i
	return count_nodes(node.left_child,i) + count_nodes(node.right_child,i)


# function that counts the number of leaves, that out tree has on all of its branches
def count_leaves(node):
	if node.is_leaf:
		return 1
	return count_leaves(node.left_child) + count_leaves(node.right_child)
print 'leaves %s' % (count_leaves(tree))
print 'nodes %s' % (count_nodes(tree,0))	

###functions for classificatio when our tree is build
def classify(row, node, variables):
	if node.is_leaf:
		return node.class_value

	if row[variables.index(node.variable)] <= node.value:
		return classify(row,node.right_child, variables)
	else:
		return classify(row,node.left_child, variables)
	
def predictions(data, node, variables):
	length=len(data.index)
	data_arr = np.array(data)
	return [classify(data_arr[i], node, variables) for i in range(length)] 
###		


###function for makig our tree as simple and effective as possible
def prune_tree(data, tree, node, variables, class_variable, best_score):
    # if node is a leaf
    if node.is_leaf == True:
        # run validate_tree on a tree with the nodes parent as a leaf with its classification
        node.parent.is_leaf = True
        node.parent.class_value = node.class_value
        new_score = validate_tree(data, tree, variables, class_variable)
        
        # if its better, change it
        if (new_score >= best_score):
            return new_score
        else:
            node.parent.is_leaf = False
            node.parent.class_value = None
            return best_score
    # if its not a leaf
    else:
        
        new_score = prune_tree(data, tree, node.right_child, variables, class_variable, best_score)

        if node.is_leaf == True:
        	return new_score

        new_score = prune_tree(data, tree, node.left_child, variables, class_variable, best_score)

        if node.is_leaf == True:
        	return new_score

        return new_score



def validate_tree(data, tree, variables, class_variable):
	data_for_test = data.copy()
	predicted = predictions(data_for_test, tree, variables)
	return sk_m.accuracy_score(np.array(data[class_variable]), np.array(predicted)) 
###





test_data = data.defined_test.copy()		
predicted = predictions(data.defined_test, tree, data.variables)



tested_arr = np.array(predicted)
test_arr = np.array(test_data[data.class_variable])


print sk_m.accuracy_score(test_arr, tested_arr)
print sk_m.roc_auc_score(test_arr, tested_arr)
print sk_m.precision_score(test_arr, tested_arr)
print sk_m.confusion_matrix(test_arr, tested_arr)

			



data_for_pruning = data.defined_test.copy()

print prune_tree(data_for_pruning, tree, tree, data.variables, data.class_variable, sk_m.accuracy_score(test_arr, tested_arr))

test_data = data.defined_test.copy()		
predicted = predictions(data_for_pruning, tree, data.variables)


tested_arr = np.array(predicted)
test_arr = np.array(test_data[data.class_variable])
	
print sk_m.accuracy_score(test_arr, tested_arr)
print sk_m.roc_auc_score(test_arr, tested_arr)
print sk_m.precision_score(test_arr, tested_arr)
print sk_m.confusion_matrix(test_arr, tested_arr)

print 'leaves : %s' % (count_leaves(tree))
print 'nodes : %s' % (count_nodes(tree,0))	
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

