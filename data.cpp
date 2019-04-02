#include "data.h"
#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <cassert>

Image::Image() {
	label = 0;
	size = 0;
}

Image::Image(vector<string> row) {
	Image::setImage(row);
}

void Image::setImage(vector<string> row) {
	size = row.size() - 1;
	content = new double[size];
	// cout << "Before: " << row[0];
	label = (int) atof(row[0].c_str());
	// cout << "  After: " << label << endl;
	for (int i = 0; i < size; ++i) {
		content[i] = atof(row[i + 1].c_str());
	}
}

void Image::printImage() {
	cout << "Image Label: " << label << endl;
	// for (int i = 0; i < size; ++i) {
	// 	cout << content[i] << " ";
	// 	if (i % 10 == 0)	cout << endl;
	// }
}

Image::~Image() {
	delete content;
}

Dataset::Dataset(string filename, string dataset_type) {
	type = dataset_type;
	// get size of the dataset
	ifstream csv(filename);
	size = 0;
	string line;
	while (getline(csv, line)) {
		size++;
	}
	// back to the begining 
	csv.clear();
	csv.seekg(0, ios::beg);
	dataset = new Image[size];
	// read CSV
	vector<string> row;
	string seg;
	int i = 0;
	while (getline(csv, line)) {
		// read an entire row
		row.clear();
		// breaking words
		stringstream s(line);
		// read every colum data of a row
		while (getline(s, seg, ',')) {
			row.push_back(seg);
		}
		dataset[i].setImage(row);
		i++;
	}
}

Dataset::~Dataset() {
	// delete dataset;
}

bool Dataset::isTrain() {
	if (type == "train") {
		return true;
	} else if (type == "test") {
		return false;
	} else {
		cout << "Error: Wrong dataset type" << endl;
		exit(-1);
	}
}

void Dataset::printDataset() {
	cout << "Dataset size: " << size << " Type:" << type << endl;
}

void Dataset::show1Image(int i) {
	if (i < size)
		cout << i << "-th ";
		dataset[i].printImage();
}