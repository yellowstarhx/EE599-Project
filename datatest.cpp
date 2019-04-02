#include "data.h"
#include <iostream>
#include <string>
using namespace std;

int main(int argc, char const *argv[])
{
	Dataset train_set = Dataset("LeNet5_train.csv", "train");
	Dataset test_set = Dataset("LeNet5_test.csv", "test");

	train_set.printDataset();
	test_set.printDataset();


	train_set.show1Image(319);

	// for (int i = 0; i < 80; i++) {
	// 	test_set.show1Image(i);
	// }

	return 0;
}