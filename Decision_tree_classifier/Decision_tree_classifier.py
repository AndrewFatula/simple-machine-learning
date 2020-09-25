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



class MyData():
	'''
	Data class with methods, needed for preprocessing and storing all the data
	'''
	def __init__(self):
		self.all_examples = None #--- all available data samples
		self.defined = None #-------- all data samples with not NA target values
		self.undefined = None #------ all data samples with NA target values
		self.variables = [] #-------- List(str) of all features in dataset
		self.target_variable = None #- (str) target variable name
		self.classes = [] #---------- (int) all unique target variables 
		self.p_initial = None #------ (float) initial probaility of one of target classes (in this case binary classification)
		self.defined_test = None #--- all test data samples with not NA target values
		self.defined_train = None #-- all train data samples with not NA target values
			
	#Read_data method reads the data and save all needed in classification data properties

	def read_data(self, directory, file):
		'''
		Method which reads data from file and store it in MyData class
		'''
		os.chdir(directory)

		self.all_examples = pd.read_csv(file, sep = ',', header = 0, dtype = 'float')
 		#all features names
		self.variables = list(data.all_examples.columns)
		#label name
		self.target_variable = self.variables.pop(-1)
		#unique class values
		classes = np.unique(data.all_examples[data.target_variable])
		self.classes = list(classes[0:2])
		#separates data with defined and undefined labels
		self.defined = data.all_examples[data.all_examples[data.variables[-1].notna()]
		self.undefined = data.all_examples[data.all_examples[data.variables[-1].isna()]
		#saves initial ration between classes in label
		self.p_initial = self.defined[self.target_variable].mean()
						   
	def fill_missing_values(self):
		'''
		Method which fills all misiing values for each variable with mean
		'''
		for variable in self.variables:
			if len(self.defined[self.defined[variable].isna()].index) > 0:
				self.defined[self.defined[variable].isna()][variable] = self.defined[self.defined[variable].isna()].mean()		

	def train_test_split(self, size):
		'''
		Method which splits all exmaples on training and test parts
		'''
		self.defined['split'] = np.random.rand(len(self.defined.index))
		self.defined_train = self.defined[self.defined['split']>size]
		self.defined_test = self.defined[self.defined['split']<=size]
		self.defined_train = self.defined_train.drop('split',1).values
		self.defined_test = self.defined_test.drop('split',1).values


data = my_data()
data.read_data('\Python27\Binary_classifiers\my\data','Data.csv')
data.fill_missing_values()
data.train_test_split(0.2)



def get_entropy(p, p_initial):
	'''
	Function which calculate information entropy considering distribution of label values before and after split
	@params:
		p - probability of class label after split
		p_initial - probability on class label before split
	'''
	if p <= p_initial:
		p_current = p/(2*p_initial)
		entropy = -(p_current*math.log(p_current+0.0001,2)+(1-p_current)*math.log(1-p_current+0.0001,2))
		
	else:
		p_current = (p+1-2*p_initial)/(2*(1-p_initial))
		entropy = -(p_current*math.log(p_current+0.0001,2)+(1-p_current)*math.log(1-p_current+0.0001,2))
	return entropy 



def get_split(data, variables, classes, p_initial, min_samples_leaf):
	'''
	function which do a split for eack node in Decision tree
	it returns split feature, split value of that feature, entropy indicator of found split and ratios of classes in label of each splitted part of the data
	return:
		split_variable - variable, considered for splitting data;
		split_value - value of split value considered for splitting data;
		entropy - information entropy of splitted data;
		p_right - ratio between label classes in data on right branch of a tree;
		p_left - ratio between label classes in data on left branch of a tree.
		
	'''
	entropy = 1
	split_variable = None
	split_value = None
	ones_r = None
	ones_l = None

	#for each feature in dataset
	for variable in variables[:-1]:
		var_idx = variables.index(variable)				   
		value_list = data[:,var_idx]
		if len(np.unique(value_list))>20:
			#finding 20 quantiles for dividing our data on given feature
			probs = [j/20.0 for j in range(1,21)]
			values = sc_st_mst.mquantiles(value_list,p robs)

		else:
			#if no values left for splitting data in given feature
			if len(np.unique(value_list))==1:
				continue 		   
			values = np.unique(value_list)	
		
		#for each quantile of the given feature
		for value in values[:-1]:
				   
			data_left =  data[data[:,var_indx] > value]
			data_rigth = data[data[:,var_indx] <= value]
			left_len = np.shape(data_left_value)[0]
			rigth_len = np.shape(data_rigth_value)[0]
			split_ratio = left_len / np.shape(data)[0]
						   
			#if hyperparameters will be surpassed after split
			if (with_len < min_samples_leaf) or (without_len < min_samples_leaf):
				continue	
			#count of data samples with first label class for each branch 		   
			ones_left = np.sum(data_left_value[variables[-1]])
			ones_rigth = np.sum(data_rigth_value[variables[-1]])

			p_left = ones_left/float(with_len)
			p_rigth = ones_rigth/float(without_len)
			### split_entropy shows how good split seperates label classes in generaly 
			split_entropy = split_ratio*get_entropy(p_left, p_initial) + (1-split_ratio)*get_entropy(p_rigth, p_initial)
			
			if split_entropy < entropy :
				p_left_chosen = p_left
				p_right_chosen = p_rigth
				entropy = split_entropy
				split_variable_idx = variables.index(variable)
				split_value = value

	if split_variable == None:
		return None		

	return split_variable, split_value, entropy, p_left_chosen, p_right_chosen


class Node():
	'''
	Node clas that saves all needed iformation about the best splits for building the decision tree	
	'''
	def __init__(self, parent, length, is_right):
		#for each split must be defined :
		#feature, where we found value for the best split
		#value for split
		#previous node, which is the parent node
		#entropy indicator of given split 
		#bool that gives as information if it is a leaf node
		#heigth of that node (current depth of a tree)
		#if it is a leaf node, class value of that leaf node
		#nodes for which current node is a parent node
		#number of rows of training data in previous split
		#the bool that gives information if current node is on the rigth branch of the previous split
		self.variable_idx = None
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
		
	
		
				
def compute_tree(data, variables, classes, p_initial, max_height=10,
				 min_samples_split=1, min_samples_leaf=1, parent=None, length=None,
				  p_current=None, is_right=False):
	'''
	function which recursively builds the decision tree (is called for building each node of a tree)
	This function is checking for each node if it is reached defined hyper_parameters, 
	if they are not reached functions finds the best split for that node
	@params:
		1) first 4 positional arguments - parameters of dataset
		2) next 4 key-arguments - hyperparams of a tree
		3) last 4 arguments - parameters passed by function itself recursively
	'''
	#defining node to find the best split or to make it a leaf node
	node = Node(parent, length, is_right)
	
	#adding depth of a tree 					   
	if node.parent == None:
		node.height = 0
		p_current = p_initial
	else:
		node.height = node.parent.height + 1
						   
	#checking hyperparameters
	if (node.length != None) and (node.length < min_samples_split):
		print 'Node_split ; complex_node'
		node.is_leaf = True
		if p_current > 0.5:
			node.class_value = classes[0]
		else:
			node.class_value = classes[1]	
		return node
																							
	#checking hyperparameters
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
		
	#for each node finding and save parameters of a split
	parameters = get_split(data, variables, classes, p_initial, min_samples_leaf)
	#if node if leaf node - saving class value
	if parameters == None:
		node.is_leaf= True
		if p_current > 0.5:
			node.class_value = classes[0]
		else:
			node.class_value = classes[1]
		return node
	
	#saving parameters returned by split function					   
	node.variable_idx = parameters[0]
	node.value = parameters[1]
	node.entropy = parameters[2]

	#get entropy for current node
	entropy = get_entropy(p_current, p_initial)
	
	#if better entropy cant be reached after next split - making leaf node and saving class labels
	if node.entropy-0.001 > entropy:
		node.is_leaf = True
		if p_current > 0.5:
			node.class_value = classes[0]
		else:
			node.class_value = classes[1]
		return node
		
	#splitting current data for the next get_split function step by the given split parameters
	data_for_right_branch = data[data[:,node.variable_idx] <= node.value]
	data_for_left_branch = data[data[:,node.variable_idx] > node.value]
	right = np.shape(data_for_right_branch)[0] 
	left = np.shape(data_for_left_branch)[0]
		
	#for each part of our splitted data call computing tree function recursively
	node.right_child = compute_tree(data_for_right_branch, variables, classes, p_initial, max_height,
					min_samples_split = min_samples_split, 
					parent = node, length = right, p_current = parameters[3], is_right = True)	
	node.left_child = compute_tree(data_for_left_branch, variables, classes, p_initial, max_height,
					min_samples_split = min_samples_split, parent = node, length = left,
				       p_current = parameters[4])

	return node




#instanciating tree model
tree = compute_tree(data.defined_train, data.variables, data.classes, data.p_initial, 20, min_samples_split = 100)


#function for counting how much node our tree has on all of its branches
def count_nodes(node,i):
	i+=1
	if node.is_leaf:
		return i
	return count_nodes(node.left_child,i) + count_nodes(node.right_child,i)


#function that counts the number of leaves, that out tree has on all of its branches
def count_leaves(node):
	if node.is_leaf:
		return 1
	return count_leaves(node.left_child) + count_leaves(node.right_child)
print 'leaves %s' % (count_leaves(tree))
print 'nodes %s' % (count_nodes(tree,0))	

def classify(row, node, variables):
	'''
	Functions used for classificatio when our tree is build
	'''
	if node.is_leaf:
		return node.class_value

	if row[variables.index(node.variable)] <= node.value:
		return classify(row,node.right_child, variables)
	else:
		return classify(row,node.left_child, variables)
	
def predictions(data, node, variables):
	''' 
	Fuction which makes predictions
	'''
	length=len(data.index)
	data_arr = np.array(data)
	return [classify(data_arr[i], node, variables) for i in range(length)] 
		


def prune_tree(data, tree, node, variables, class_variable, best_score):
	'''
	function for makig our tree as simple and effective as possible
	'''
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
	# if its not a leaf call it recursively for each node
	else:

		new_score = prune_tree(data, tree, node.right_child, variables, class_variable, best_score)
		if node.is_leaf == True:
			return new_score

		new_score = prune_tree(data, tree, node.left_child, variables, class_variable, best_score)
		if node.is_leaf == True:
			return new_score

		return new_score


def validate_tree(data, tree, variables, class_variable):
	'''
	Function which validated performance of built tree
	'''
	data_for_test = data.copy()
	predicted = predictions(data_for_test, tree, variables)
	return sk_m.accuracy_score(np.array(data[class_variable]), np.array(predicted)) 


test_data = data.defined_test.copy()		
predicted = predictions(data.defined_test, tree, data.variables)

tested_arr = np.array(predicted)
test_arr = np.array(test_data[data.target_variable])

#printing evaluation results of trained tree model
print sk_m.accuracy_score(test_arr, tested_arr)
print sk_m.roc_auc_score(test_arr, tested_arr)
print sk_m.precision_score(test_arr, tested_arr)
print sk_m.confusion_matrix(test_arr, tested_arr)

data_for_pruning = data.defined_test.copy()

#prune and test prunned tree
print( prune_tree(data_for_pruning, tree, tree, data.variables, data.class_variable, sk_m.accuracy_score(test_arr, tested_arr)) )

test_data = data.defined_test.copy()		
predicted = predictions(data_for_pruning, tree, data.variables)

tested_arr = np.array(predicted)
test_arr = np.array(test_data[data.class_variable])

#printing evaluation results of trained and prunned tree model
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
print(all_time_sec)	

