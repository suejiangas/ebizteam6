# import csv
import random
import math
import operator
import arff, numpy as np

TRAIN_FILE_NAME_A = 'trainProdSelection.arff'
TEST_FILE_NAME_A = 'testProdSelection.arff'
# CONTINUEOUS_ATTRIBUTES_FLAG_A  = [False, False, True, True, True, True]
# SYMBOL_ATTRIBUTES_FLAG_A = [True, True, False, False, False, False]

# SIMILARITY_MATRIX_CUSTOMER_TYPE = array([])

def load_data_set(filename, split, train_set=[] , test_set=[], attributes=[]):
	with open(filename, 'rb') as f:
		dataset = arff.load(f)
		# data = np.array(dataset['data'])

		data = list(dataset['data'])
		attr = list(dataset['attributes'])
		
		for x in range(len(data) - 1):
			if random.random() < split:
				train_set.append(data[x])
			else:
				test_set.append(data[x])

		"""
		In order to make the program workable for both Problem A and B, 
		the attributes information of data are return here.

		For .arff file, the format of each attribute colume will be (colume_name, colume_type).

		If the columne type is 'REAL', it is a numeric attribute, we treat it as continuous value
		and will be normalized into range [0, 1] before calculating distance.

		Otherwise, the columne will be treated as a symbolic attribute. 
		""" 
		for x in range(len(attr)):
			attributes.append(attr[x])

# def is_continuous_value(column_index, attributes):
# 	return attributes[column_index][1] == 'REAL'
# 	# return CONTINUEOUS_ATTRIBUTES_FLAG_A[column_index]

def calculate_min_max_values(train_set, attributes, min_values, max_values):
	length = len(min_values)
	for x in range(length):
		if attributes[x][1] == 'REAL':
			min_values[x] = min(train_set, key=lambda i: float(i[x]))[x]
			max_values[x] = max(train_set, key=lambda i: float(i[x]))[x]

def normalize_data(data_set, attributes, min_values, max_values):
	"""
	For continuous value: v_norm = (v_actual - v_min) / (v_max - v_min)
	"""
	length = len(min_values)
	for x in range(len(data_set)):
		for y in range(length):
			if attributes[y][1] == 'REAL':
				data_set[x][y] = (data_set[x][y] - min_values[y]) / (max_values[y] - min_values[y])


def normalize_train_and_test_data(train_set, test_set, attributes):
	min_values = [0] * len(test_set[0])
	max_values = [0] * len(test_set[0])

	calculate_min_max_values(train_set, attributes, min_values, max_values)
	normalize_data(train_set, attributes, min_values, max_values)
	normalize_data(test_set, attributes, min_values, max_values)

def euclidean_distance(instance1, instance2, attributes, length):
	distance = 0

	for x in range(length):
		if attributes[x][1] == 'REAL':
			distance += abs(instance1[x] - instance2[x])**2
		else:
			"""
			For simplicity, the similarity of two symbolic attributes set to 1 if they are equal with each other,
			and set to 0 if not. This is according to similarity matrix given by the professor.

			In the future, if the similarity matrix got changed, the similarity may need to read and abstract directly
			from the .xlsx file, so refactor may needed here.
			"""
			similarity = int(instance1[x] == instance2[x])
			distance += 1 - similarity
	return 1/math.sqrt(distance)

 
def get_k_neighbors(train_set, test_instance, attributes, k):
	distances = []
	
	for x in range(len(train_set)):
		dist = euclidean_distance(test_instance, train_set[x], attributes, len(test_instance))
		distances.append((train_set[x], dist))

	distances.sort(key=operator.itemgetter(1), reverse=True)
	
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x])
	return neighbors
 
def predicate_distance_weights(neighbors):
	class_votes = {}
	for x in range(len(neighbors)):
		dist = neighbors[x][-1]
		label = neighbors[x][-2]
		class_votes[label] += 1 / (dist**2 + 1)
	labels, votes = zip(*class_votes.most_common())

	winner = class_votes.most_common(1)[0][0]
	votes4winner = class_votes.most_common(1)[0][1]
	return winner

def predicate_class(neighbors):
	class_scores = {}
	for x in range(len(neighbors)):
		label = neighbors[x][0][-1]
		# class_scores[label] += neighbors[x][1]
		if label in class_scores:
			class_scores[label] += neighbors[x][1]
		else:
			class_scores[label] = neighbors[x][1]
	sorted_scores = sorted(class_scores.iteritems(), key=operator.itemgetter(1), reverse=True)
	# print sorted_scores
	return sorted_scores[0][0]
 
def get_accuracy(test_labels, predictions):
	correct = 0
	for x in range(len(test_labels)):
		if test_labels[x] == predictions[x]:
			correct += 1
	return (correct/float(len(test_labels))) * 100.0
	
def main():
	# prepare data
	train_set=[]
	test_set=[]
	attributes=[]
	split = 0.67
	load_data_set(TRAIN_FILE_NAME_A, split, train_set, test_set, attributes)

	"""
	Split the test data into test_set and test_labels.

	test_labels contains only the class of each test instance 
	and will be used in accuracy calculation
	
	test_set contains the values need to be used for distance calculation except label
	"""
	test_labels = [x[-1] for x in (x for x in test_set)]
	test_set = [x[:-1] for x in (x for x in test_set)]

	# normalize continuous data
	normalize_train_and_test_data(train_set, test_set, attributes)

	k = 3
	predictions = []

	# calculate prediction
	for x in range(len(test_set)):
		neighbors = get_k_neighbors(train_set, test_set[x], attributes, k)
		# print neighbors
		result = predicate_class(neighbors)
		predictions.append(result)
		print('> predicted=' + result + ', actual=' + test_labels[x])

	# calculate accuracy
	accuracy = get_accuracy(test_labels, predictions)
	print('Accuracy: ' + repr(accuracy) + '%')
	

if __name__ == '__main__':
    main()