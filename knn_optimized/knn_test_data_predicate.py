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


def load_data_set(filename, attributes=[], is_train=True):
    with open(filename, 'rb') as f:
        # print(list(f))
        dataset = arff.load(f)
        data = list(dataset['data'])
        # random.shuffle(data)
        if is_train:
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
    length = len(attributes) - 1
    min_values = [0] * length
    max_values = [0] * length

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
    if distance == 0:
        return 1
    return 1 / math.sqrt(distance)


def get_k_neighbors(train_set, test_instance, attributes,
                    k, weights):
    distances = []

    for x in range(len(train_set)):
        dist = euclidean_distance(test_instance,
                                  train_set[x], attributes,
                                  len(attributes)-1,
                                  weights)
        distances.append((train_set[x], dist))

    distances.sort(key=operator.itemgetter(1), reverse=True)

    neighbors = []
    for x in range(k):
        neighbors.append(distances[x])
    return neighbors

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

def knn_job(log_name, train_set, test_set, attributes, k, validation_fold,
            weights, old_data):
    # prepare data
    accuracies = []
    for fold in range(validation_fold):
        """
        Split the test data into test_set and test_labels.

        test_labels contains only the class of each test instance
        and will be used in accuracy calculation

        test_set contains the values need to be used for distance calculation except label
        """

        # normalize continuous data
        normalize_continuous_data(train_set, test_set,
                                  attributes)


        # convert symbolic data
        convert_symbolic_data(train_set, attributes)
        convert_symbolic_data(test_set, attributes)

        predictions = []
        log_file = open(log_name, "a+")

        # calculate prediction
        for x in range(len(test_set)):
            neighbors = get_k_neighbors(train_set,
                                        test_set[x],
                                        attributes, k,
                                        weights)
            if isBReal:
                result = 0.0;
                for i in range(len(neighbors)):
                    neighbor = neighbors[i][0]
                    result = result + neighbor[len(neighbor) - 1]
                result = result / len(neighbors)
            else:
                result = predicate_class(neighbors)
                predictions.append(result)
            log_file.write('> attributes= ' + str(old_data[x])[1:-1] + ', predicted=' + str(result))
            log_file.write('\r\n')
            # print('> attributes= ' + str(old_data[x])[1:-1] + ', predicted=' + str(result))
        log_file.close()

def main(argv):
    if len(argv) < 3:
        print("Invalid input, the input should be as : python knn.py {train_filename} {test_filename}")
        return

    train_filename = argv[1]
    test_filename = argv[2]
    log_name = "knn_log.txt"

    global isBReal
    if train_filename.split('.')[1] == 'real':
        isBReal = True
    else:
        isBReal = False

    attributes = []

    validation_fold = 1
    train_data = load_data_set(train_filename, attributes, True)
    test_data = load_data_set(test_filename, attributes, False)

    old_data = []
    for x in range(len(test_data)):
        old_data.append(copy.copy(test_data[x]))

    global attr_weights
    if len(attributes) == 9:
        # problem b
        k = 3
        attr_weights = [2.0, 2.0, 8.0, 8.0, 1.0, 1.0, 2.0, 4.0]
    else:
        # problem a
        attr_weights = [1.0, 0.5, 1.0, 16.0, 4.0, 0.5]
        k = 5

    knn_job(log_name, train_data, test_data, attributes, k,
                         validation_fold, attr_weights, old_data)

    print("done")


if __name__ == '__main__':
    main(sys.argv)

# for problem a, best value is k = 5, weights = [1.0,
# 0.5, 1.0, 16.0, 4.0, 0.5, 0.125]
# for problem b, best value is k = 3, weights = [2.0,
# 2.0, 8.0, 8.0, 1.0, 1.0, 2.0, 4.0, 2.0]
