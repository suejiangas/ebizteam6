import sys
import xlrd
import random
import math
import operator
import arff, numpy as np

TRAIN_FILE_NAME_A = 'trainProdSelection.arff'
TEST_FILE_NAME_A = 'testProdSelection.arff'

TRAIN_FILE_NAME_B = 'trainProdIntro.binary.arff'
TEST_FILE_NAME_B = 'testProdIntro.binary.arff'

SIMILARITY_MATRIX_FILE_NAME = 'similaritymatrixUpdatedVersion.xls'

SIMILARITY_MATRIX = {'Type' : ([[1,0,0,0,0],
								[0,1,0,0,0],
								[0,0,1,0,0],
								[0,0,0,1,0],
								[0,0,0,0,1]]),
					 'LifeStyle' : ([[1,0,0,0],
					 				 [0,1,0,0],
					 				 [0,0,1,0],
					 				 [0,0,0,1]]),

					 'Service_type' : ([1, 0, 0.1, 0.3, 0.2],
					 				   [0, 1, 0, 0, 0],
					 				   [0.1, 0, 1, 0.2, 0.2],
					 				   [0.3, 0, 0.2, 1, 0.2],
					 				   [0.2, 0, 0.2, 0.1, 1]),

					 'Customer': ([1, 0.2, 0.1, 0.2, 0],
					 			  [0.2, 1, 0.2, 0.1, 0],
					 			  [0.1, 0.2, 1, 0.1, 0],
					 			  [0.2, 0.1, 0.1, 1, 0],
					 			  [0, 0, 0, 0, 1]),

					 'Size': ([1, 0.1, 0],
					 	      [0.1, 1, 0.1],
					 	      [0, 0.1, 1]),

					 'Promotion': ([1, 0.8, 0, 0],
					 			   [0.8, 1, 0.1, 0.5],
					 			   [0, 0.1, 1, 0.4],
					 			   [0, 0.5, 0.4, 1])
					 }
SYMBOLIC_MAP = {'Type' : {'student':0, 'engineer': 1, 'librarian': 2, 'professor' : 3, 'doctor': 4},
				'LifeStyle': {'spend<<saving': 0, 'spend<saving': 1, 'spend>saving':2, 'spend>>saving':3},
				'Service_type': {'Loan': 0, 'Bank_Account': 1, 'CD': 2, 'Mortgage':3, 'Fund':4},
				'Customer' : {'Business':0, 'Professional':1, 'Student':2, 'Doctor':3, 'Other':4},
				'Size': {'Small':0, 'Medium':1, 'Large':2},
				'Promotion': {'Full':0, 'Web&Email':1, 'Web':2, 'None':3}
				}


def load_data_set(filename, split, train_set=[] , test_set=[], attributes=[]):
	with open(filename, 'rb') as f:
		dataset = arff.load(f)
		# data = np.array(dataset['data'])

		data = list(dataset['data'])
		attr = list(dataset['attributes'])
		# print attr
		
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


def normalize_continuous_data(train_set, test_set, attributes):
	min_values = [0] * len(test_set[0])
	max_values = [0] * len(test_set[0])

	calculate_min_max_values(train_set, attributes, min_values, max_values)
	normalize_data(train_set, attributes, min_values, max_values)
	normalize_data(test_set, attributes, min_values, max_values)


def convert_symbolic_data(data_set, attributes):
	"""
	For each symbolic value, convert the symbol into its index according attribute definition.
	E.g. For problem B, if the 'Service_type' is 'Loan', the converted value should be 1.
		 The reason is, in attribute definition, the value range of 'Service_type' is:
		 ['Fund', 'Loan', 'CD', 'Bank_Account', 'Mortgage'], and the index of 'Loan' is 1
	"""
	for x in range(len(data_set)):
		for y in range(len(attributes) - 1):
			if attributes[y][1] != 'REAL':
				data_set[x][y] = SYMBOLIC_MAP[attributes[y][0]][data_set[x][y]]

				# for i in range(len(attributes[y][1])):
				# 	if data_set[x][y] == attributes[y][1][i]:
				# 		data_set[x][y] = i;
				# 		break;

def euclidean_distance(instance1, instance2, attributes, length):
	distance = 0

	for x in range(length):
		if attributes[x][1] == 'REAL':
			distance += abs(instance1[x] - instance2[x])**2
		else:
			similarity = SIMILARITY_MATRIX[attributes[x][0]][instance1[x]][instance2[x]]
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
	

# def read_similarity_matrix(problem):
# 	data = xlrd.open_workbook(SIMILARITY_MATRIX_FILE_NAME)
# 	if problem == 'A':
# 		 table = data.sheets()[0]
# 		 # table = data.sheet_by_name('Task A')
# 		 print table 
# 	else :
# 		 table = data.sheets()[1]	
# 		 print table 	

def main(argv):
	if argv[1] == 'A':
		filename = TRAIN_FILE_NAME_A
	elif argv[1] == 'B':
		filename = TRAIN_FILE_NAME_B	
	else :
		print "Invalid input"
		return 

	# prepare data
	train_set=[]
	test_set=[]
	attributes=[]
	split = 0.67
	load_data_set(filename, split, train_set, test_set, attributes)
	print attributes
	"""
	Split the test data into test_set and test_labels.

	test_labels contains only the class of each test instance 
	and will be used in accuracy calculation
	
	test_set contains the values need to be used for distance calculation except label
	"""
	test_labels = [x[-1] for x in (x for x in test_set)]
	test_set = [x[:-1] for x in (x for x in test_set)]

	# normalize continuous data
	normalize_continuous_data(train_set, test_set, attributes)

	# convert symbolic data
	convert_symbolic_data(train_set, attributes)
	convert_symbolic_data(test_set, attributes)

	k = 3
	predictions = []

	# calculate prediction
	for x in range(len(test_set)):
		neighbors = get_k_neighbors(train_set, test_set[x], attributes, k)
		# print neighbors
		result = predicate_class(neighbors)
		predictions.append(result)
		print('> predicted=' + result + ', actual=' + test_labels[x])

	# # calculate accuracy
	accuracy = get_accuracy(test_labels, predictions)
	print('Accuracy: ' + repr(accuracy) + '%')
	

if __name__ == '__main__':
    main(sys.argv)