


#####

Here you can find the implementation from scratch of decision tree regressor algorithm.

regression_tree.py - regression tree model, written from scratch using popular Python frameworks Numpy and Pandas,
model trained and tested on artifical generated data using pseudo-random values generator.

compute_tree - function that recursively builds regression tree, 
the main principle of building the three is: finding specific value of one of the given features, that variation of labels of twoo splitted on those value parts is minimal.

#####

Regression_tree_example.py - regression tree model, written almost from scratch(except linear regression predictor) using Python frameworks Numpy, Pandas and sklearn,
model trained and tested on AirQualitUCI data, description of the data you can find here https://archive.ics.uci.edu/ml/datasets/Air+quality

The target of the analyse is to predict air temperature, given the chemical composition and time and date of the observation.

compute_tree - function that builds regression tree on the date and time features, 
and then for data on each leaf node apply sklearn LinearRegression predictor, trained on chemical composition features.

Mean absolute error of predictions for Regression_tree_example model is 0.65 .

#####

