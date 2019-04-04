#include "KNN.h"
#include <stdlib.h>

int main(int argc, char const *argv[])
{
	int k;
	string train_csv, test_csv;
	if (argc < 2) {
		// default k
		k = 5;
	} else {
		k = atoi(argv[1]);
	}
	if (argc < 3) {
		train_csv = "LeNet5_train.csv";
	} else {
		train_csv = argv[2];
	}
	if (argc < 4) {
		test_csv = "LeNet5_test.csv";
	} else {
		test_csv = argv[2];
	}

	KNN model;
	model.set_train_data(train_csv);
	model.set_test_data(test_csv);
	model.predict_all(k);

	return 0;
}