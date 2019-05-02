#include<vector>
#include<iostream>
#include<fstream>
#include<string>
#include<sstream>
 
using namespace std;

// class Image{
// 	int label;
// 	double* content;
// public:
// 	Image();
// 	~Image();
// };

int main(int argc, char const *argv[]) {
	int label;

	ifstream testset_csv("LeNet5_test.csv");

	int testset_size = 1;

	vector<string> row;
	string line, word;
	int count = 0;
	while (getline(testset_csv, line)) {
		// read an entire row
		row.clear();
		// show origin line
		// cout << line << endl;
		// breaking words
		stringstream s(line);
		// read every colum data of a row
		int element_count = 0;
		while (getline(s, word, ',')) {
			row.push_back(word);
			element_count++;
			// cout << word << endl;
		}
		cout << element_count << endl;
		cout << row.size() << endl;
		count++;
		if (count == testset_size) {
			break;
		}
	}

	label = stoi(row[0]);
	cout << "The first test image label: " << label << endl << endl;
	// for (int i = 1; i <= (28*28); i++) {
	// 	cout << row[i];
	// 	if (i % 28 == 0) {
	// 		cout << endl;
	// 	}
	// } 

	return 0;
}