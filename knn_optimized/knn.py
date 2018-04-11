import sys
import random
import math
import operator
import arff
import copy

TRAIN_FILE_NAME_A = 'trainProdSelection.arff'
TEST_FILE_NAME_A = 'testProdSelection.arff'

TRAIN_FILE_NAME_B = 'trainProdIntro.binary.arff'
TEST_FILE_NAME_B = 'testProdIntro.binary.arff'

SIMILARITY_MATRIX_FILE_NAME = 'similaritymatrixUpdatedVersion.xls'

SIMILARITY_MATRIX = {'Type': ([[1, 0, 0, 0, 0],
                               [0, 1, 0, 0, 0],
                               [0, 0, 1, 0, 0],
                               [0, 0, 0, 1, 0],
                               [0, 0, 0, 0, 1]]),
                     'LifeStyle': ([[1, 0, 0, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]]),

                     'Service_type': ([1, 0, 0.1, 0.3, 0.2],
                                      [0, 1, 0, 0, 0],
                                      [0.1, 0, 1, 0.2, 0.2],
                                      [0.3, 0, 0.2, 1, 0.2],
                                      [0.2, 0, 0.2, 0.1,
                                       1]),

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
SYMBOLIC_MAP = {
    'Type': {'student': 0, 'engineer': 1, 'librarian': 2,
             'professor': 3, 'doctor': 4},
    'LifeStyle': {'spend<<saving': 0, 'spend<saving': 1,
                  'spend>saving': 2, 'spend>>saving': 3},
    'Service_type': {'Loan': 0, 'Bank_Account': 1, 'CD': 2,
                     'Mortgage': 3, 'Fund': 4},
    'Customer': {'Business': 0, 'Professional': 1,
                 'Student': 2, 'Doctor': 3, 'Other': 4},
    'Size': {'Small': 0, 'Medium': 1, 'Large': 2},
    'Promotion': {'Full': 0, 'Web&Email': 1, 'Web': 2,
                  'None': 3}
}


def load_data_set(filename, attributes=[]):
    with open(filename, 'rb') as f:
        # print(list(f))
        dataset = arff.load(f)
        # data = np.array(dataset['data'])
        data = list(dataset['data'])
        random.shuffle(data)
        attr = list(dataset['attributes'])

        for x in range(len(attr)):
            attributes.append(attr[x])

    return data


def split_test_and_train_set(old_data,
                             test_set_start_percentage,
                             test_set_end_percentage,
                             train_set=[],
                             test_set=[]):
    data = []
    for i in range(len(old_data)):
        data.append(copy.copy(old_data[i]))

    for x in range(int(len(
            data) * test_set_start_percentage)):
        train_set.append(data[x])

    for x in range(int(len(
            data) * test_set_start_percentage),
                   int(len(
                       data) * test_set_end_percentage)):
        test_set.append(data[x])

    for x in range(int(len(
            data) * test_set_end_percentage),
                   len(data)):
        train_set.append(data[x])

    """
		In order to make the program workable for both Problem A and B,
		the attributes information of data are return here.
		For .arff file, the format of each attribute colume will be (colume_name, colume_type).
		If the columne type is 'REAL', it is a numeric attribute, we treat it as continuous value
		and will be normalized into range [0, 1] before calculating distance.
		Otherwise, the column will be treated as a symbolic attribute.
		"""


# def is_continuous_value(column_index, attributes):
# 	return attributes[column_index][1] == 'REAL'
# 	# return CONTINUEOUS_ATTRIBUTES_FLAG_A[column_index]

def calculate_min_max_values(train_set, attributes,
                             min_values, max_values):
    length = len(min_values)
    for x in range(length):
        if attributes[x][1] == 'REAL':
            min_values[x] = \
                min(train_set, key=lambda i: float(i[x]))[x]
            max_values[x] = \
                max(train_set, key=lambda i: float(i[x]))[x]


def normalize_data(data_set, attributes, min_values,
                   max_values):
    """
	For continuous value: v_norm = (v_actual - v_min) / (v_max - v_min)
	"""
    length = len(min_values)
    for x in range(len(data_set)):
        for y in range(length):
            if attributes[y][1] == 'REAL':
                data_set[x][y] = (data_set[x][y] -
                                  min_values[y]) / (
                                         max_values[y] -
                                         min_values[y])


def normalize_continuous_data(train_set, test_set,
                              attributes):
    min_values = [0] * len(test_set[0])
    max_values = [0] * len(test_set[0])

    calculate_min_max_values(train_set, attributes,
                             min_values, max_values)
    normalize_data(train_set, attributes, min_values,
                   max_values)
    normalize_data(test_set, attributes, min_values,
                   max_values)


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
                data_set[x][y] = \
                    SYMBOLIC_MAP[attributes[y][0]][
                        data_set[x][y]]

            # for i in range(len(attributes[y][1])):
            # 	if data_set[x][y] == attributes[y][1][i]:
            # 		data_set[x][y] = i;
            # 		break;


def euclidean_distance(instance1,
                       instance2, attributes,
                       length, weights):
    distance = 0

    for x in range(length):
        if attributes[x][1] == 'REAL':
            distance += (abs(
                instance1[x] - instance2[x]) * weights[x]) \
                        ** 2
        else:
            similarity = \
                SIMILARITY_MATRIX[attributes[x][0]][
                    instance1[x]][instance2[x]]
            distance += (1 - similarity) * weights[x] * \
                        weights[x]
    return 1 / math.sqrt(distance)


def get_k_neighbors(train_set, test_instance, attributes,
                    k, weights):
    distances = []

    for x in range(len(train_set)):
        dist = euclidean_distance(test_instance,
                                  train_set[x], attributes,
                                  len(test_instance),
                                  weights)
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
        class_votes[label] += 1 / (dist ** 2 + 1)
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
    sorted_scores = sorted(class_scores.iteritems(),
                           key=operator.itemgetter(1),
                           reverse=True)
    # print sorted_scores
    return sorted_scores[0][0]


def get_accuracy(test_labels, predictions):
    correct = 0
    for x in range(len(test_labels)):
        if test_labels[x] == predictions[x]:
            correct += 1
    return (correct / float(len(test_labels))) * 100.0


# def read_similarity_matrix(problem):
# 	data = xlrd.open_workbook(SIMILARITY_MATRIX_FILE_NAME)
# 	if problem == 'A':
# 		 table = data.sheets()[0]
# 		 # table = data.sheet_by_name('Task A')
# 		 print table
# 	else :
# 		 table = data.sheets()[1]
# 		 print table


def knn_job(log_name, data, attributes, k, validation_fold,
            weights):
    # prepare data
    accuracies = []
    for fold in range(validation_fold):
        train_set = []
        test_set = []
        test_set_start_percentage = (
                                            1.0 / validation_fold) * (
                                        fold)
        test_set_end_percentage = (
                                          1.0 / validation_fold) * (
                                          fold + 1)

        split_test_and_train_set(data,
                                 test_set_start_percentage,
                                 test_set_end_percentage,
                                 train_set,
                                 test_set, )

        """
        Split the test data into test_set and test_labels.

        test_labels contains only the class of each test instance
        and will be used in accuracy calculation

        test_set contains the values need to be used for distance calculation except label
        """
        test_labels = [x[-1] for x in (x for x in test_set)]
        test_set = [x[:-1] for x in (x for x in test_set)]

        # normalize continuous data
        normalize_continuous_data(train_set, test_set,
                                  attributes)


        # convert symbolic data
        convert_symbolic_data(train_set, attributes)
        convert_symbolic_data(test_set, attributes)

        predictions = []

        # calculate prediction
        for x in range(len(test_set)):
            neighbors = get_k_neighbors(train_set,
                                        test_set[x],
                                        attributes, k,
                                        weights)
            # print neighbors;
            result = predicate_class(neighbors)
            predictions.append(result)
            # print('> predicted=' + result + ', actual=' +
                  # test_labels[x])

        # # calculate accuracy
        accuracy = get_accuracy(test_labels, predictions)
        # print('Accuracy: ' + repr(accuracy) + '%')
        accuracies.append(accuracy)

    final_accuracy = sum(accuracies) / len(accuracies)

    log_file = open(log_name, "a+")

    log_file.write('%10s' % repr(k))
    log_file.write('%20s' % repr(validation_fold))
    log_file.write('%65s' % repr(weights))
    log_file.write('%25s' % repr(final_accuracy) + '%\r\n')
    log_file.close()

    return final_accuracy


def main(argv):
    if argv[1] == 'A':
        filename = TRAIN_FILE_NAME_A
        log_name = "A_optimization_log.txt"
    elif argv[1] == 'B':
        filename = TRAIN_FILE_NAME_B
        log_name = "B_optimization_log.txt"
    else:
        print("Invalid input")
        return
    attributes = []
    k = 5
    validation_fold = 5
    data = load_data_set(filename, attributes)


    global attr_weights
    # a: attr_weights = [0.25, 0.125, 0.5, 4.0, 1.0, 0.25,
    # 0.125] 91.37
    # attr_weights = [2.0, 2.0, 8.0, 8.0, 1.0, 1.0, 2.0,
    # 4.0, 2.0]
    # a: attr_weights = [8.0, 2.0, 8.0, 64.0, 32.0, 4.0,
    # 1.0] 91.43
    # [1.0, 0.5, 1.0, 16.0, 4.0, 0.5, 0.125] 91.92
    # [1.0, 0.5, 1.0, 16.0, 4.0, 1.0, 0.125] 91.81
    # [1.0, 0.5, 1.0, 8.0, 4.0, 1.0, 0.125] 91.51
    # [1.0, 0.5, 1.0, 8.0, 4.0, 0.5, 0.125] 91.45
    attr_weights = [1.0, 0.5, 1.0, 16.0, 4.0, 0.5, 0.125]

    log_file = open(log_name, "a+")

    log_file.write('%10s' % 'K-neighbor')
    log_file.write('%20s' % 'K-fold')
    log_file.write('%65s' % 'Weights')
    log_file.write('%25s' % 'Accuracy\r\n')
    log_file.close()

    # for i in range(len(attr_weights)):
    #     # attr_weights[i] = attr_weights[i] / 2.0
    #     global best_result
    #     global curr_result
    #     best_result = 0.0
    #     curr_result = 0.01
    #     while best_result < curr_result:
    #         best_result = curr_result
    #         curr_result = knn_job(log_name, data,
    #                               attributes,
    #                               k, validation_fold,
    #                               attr_weights)
    #         attr_weights[i] = attr_weights[i] * 2.0
    #     attr_weights[i] = attr_weights[i] / 2.0
    #     attr_weights[i] = attr_weights[i] / 2.0
    #
    #     log_file = open(log_name, "a+")
    #     log_file.write(
    #         '------------------------------------------------------------------\r\n')
    #     log_file.close()

    global results
    results = []
    for i in range(500):
        random.shuffle(data)
        result = knn_job(log_name, data, attributes, k,
                         validation_fold, attr_weights)
        results.append(result)
    log_file = open(log_name, "a+")
    log_file.write("Average Accuracy: " + repr(sum(
        results)/len(results)))
    log_file.write(
        '------------------------------------------------------------------\r\n')
    log_file.close()

    print("done")


if __name__ == '__main__':
    main(sys.argv)

# for problem a, best value is k = 5, weights = [1.0,
# 0.5, 1.0, 16.0, 4.0, 0.5, 0.125]
# for problem b, best value is k = 3, weights = [2.0,
# 2.0, 8.0, 8.0, 1.0, 1.0, 2.0, 4.0, 2.0]
