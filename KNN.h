// KNN.h
#ifndef KNN_H
#define KNN_H

#include "data.h"
#include <string>
#include <iostream>
#include <cmath>
#include <utility>
#include <map>
#include <cassert>
using namespace std;

class KNN {
	Dataset* train_data;
	Dataset* test_data;
	int k; 
public:
	KNN();
	~KNN();
	void set_k(int k);
	void set_train_data(string filename);
	void set_test_data(string filename);
	int predict(Image img);
	int predict(Image img, int k);
	void predict_all();
	void predict_all(int k);
	double compute_distence(Image img_test, Image img_train);
	double compute_distence(double* ct1, double* ct2, int size);
};

#endif