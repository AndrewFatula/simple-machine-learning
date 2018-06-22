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
		



	def read_data(self, directory, file):
		os.chdir(directory)

		self.all_examples = pd.read_csv(file, sep = ',', header = 0, dtype = 'float', na_values = '?')
		variables = list(data.all_examples.columns)
 
		self.variables = copy.copy(variables)
		self.variables.remove(variables[-1])
	
		self.class_variable = variables[-1]
		
		classes = np.unique(data.all_examples[data.class_variable])
		self.classes = list(classes[0:3])

		self.defined = data.all_examples[data.all_examples[data.class_variable].notna()]
		self.undefined = data.all_examples[data.all_examples[data.class_variable].isna()]
		
		self.p_initial = (len(self.all_examples[self.all_examples[self.class_variable]==self.classes[0]].index)/
			float(len(self.all_examples.index)))

	def fill_missing_values(self):
		for variable in self.variables:
			if 	len(self.defined[self.defined[variable].isna()].index) > 0:
				p_na = (len(self.defined[(self.defined[variable].isna()) & (self.defined[self.class_variable]==self.classes[1])].index)/
					float(len(self.defined[self.defined[variable].isna()].index)))
				self.defined[variable] = self.defined[variable].fillna(p_na*np.mean(self.defined[variable]))		

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

def get_entropy(p, p_initial):
	if p <= p_initial:
		p_current = p/(2*p_initial)
		entropy = -(p_current*math.log(p_current+0.0001,2)+(1-p_current)*math.log(1-p_current+0.0001,2))
		
	else:
		p_current = (p+1-2*p_initial)/(2*(1-p_initial))
		entropy = -(p_current*math.log(p_current+0.0001,2)+(1-p_current)*math.log(1-p_current+0.0001,2))
	return entropy 


def get_split(data, variables, class_variable, classes, p_initial, min_samples_leaf):

	entropy = 1
	split_variable = None
	split_value = None
	ones_r = None
	ones_l = None


	for variable in variables:
		value_list = data[variable]

		if len(np.unique(value_list))>20:
			
			probs = [j/20.0 for j in range(1,21)]
			values = sc_st_mst.mquantiles(value_list,probs)
		else:
			if len(np.unique(value_list))==1:
				continue
			values = np.unique(value_list)	
			
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
			### split_entropy shows how good split seperates class_values in generaly 
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
class Node():
	def __init__(self, parent, length, is_right):
		
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
		
	
		
		
		
def compute_tree(data, variables, class_variable, classes, p_initial, max_height,
				 min_samples_split = 1, min_samples_leaf = 1, parent=None, length = None,
				  p_current = None, is_right = False):
	
	print '111'
	node = Node(parent, length, is_right)

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
	print entropy
	

	if node.entropy-0.001 > entropy:
		print 'Entropy ; complex_node'
		node.is_leaf = True
		if p_current > 0.5:
			node.class_value = classes[0]
		else:
			node.class_value = classes[1]
		return node
		

	data_for_right_branch = data[data[node.variable] <= node.value]
	data_for_left_branch = data[data[node.variable] > node.value]
	right = len(data_for_right_branch.index) 
	left = len(data_for_left_branch.index)
		

	node.right_child = compute_tree(data_for_right_branch, variables, class_variable, classes, p_initial, max_height,
										min_samples_split = min_samples_split, parent = node, length = right, p_current = parameters[3], is_right = True)	
	node.left_child = compute_tree(data_for_left_branch, variables, class_variable, classes, p_initial, max_height,
										min_samples_split = min_samples_split, parent = node, length = left, p_current = parameters[4])
	


	return node

tree = compute_tree(data.defined_train, data.variables, data.class_variable, data.classes, data.p_initial, 20, min_samples_split = 100)

def count_nodes(node,i):
	i+=1
	if node.is_leaf:
		return i
	return count_nodes(node.left_child,i) + count_nodes(node.right_child,i)



def count_leaves(node):
	if node.is_leaf:
		return 1
	return count_leaves(node.left_child) + count_leaves(node.right_child)
print 'leaves %s' % (count_leaves(tree))
print 'nodes %s' % (count_nodes(tree,0))	

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

