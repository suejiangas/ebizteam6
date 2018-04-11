# TEAM6 Task 11 Data Mining - CRM KNN Implementation

## Prerequisite:
1. Install "liac-arff" package by run: pip install liac-arff. We use liac-arff package to load arff file
2. For the test data file, please use the test file we provided. The reason is, the test data file provided by professor, including testProdSelection.arff and testProdIntro.real.arff have an invalid data format for "liac-arff" - the two data file have "Label" in their "@ATTRIBUTES" section but not empty column in the "@DATA" section so that a format mistach will happen when using "liac-arff" to load these two data file.
   Therefore, we removed the "Label" Attribute label from these two original arff file in order to load the data successfully.

To get the predicton result, please run the program with command as follow:

python knn_test_data_predicate.py {train_data_file_name} {test_data_file_name}

Three argumengs need to be provided:
argv[0] --> file name of python program
argv[1] --> file name of train data
argv[2] --> file name of test data

For example, by running command: python knn_test_predicate.py trainProdSelection.arff testProdSelection.arff, the predication result of problem A will be output into the log file.
