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

class ML { 	// machine learning methods, abstract class
public:
	virtual void set_train_data(string filename) {}
	virtual void set_test_data(string filename) {}
	virtual int predict(Image img) {}
	virtual void predict_all() {}
};

class MATH {
public:
	// math operations
	double add(double a, double b);
	double minus(double a, double b);
	double power(double a, double b);
	virtual double distL2(double* ct1, double* ct2, int size);
};

typedef double (MATH::*pfunArray) (double, double);

class MATH_IMG : public MATH {	// the methods to compute the distance between images
public:
	double distL2(double* ct1, double* ct2, int size);
};

class KNN : public ML {
	Dataset* train_data;
	Dataset* test_data;
	int k; 
public:
	KNN();
	~KNN();
	void set_k(int k);
	void set_train_data(string filename);
	void set_test_data(string filename);
	int predict(Image& img);
	int predict(Image img, int k);
	void predict_all();
	void predict_all(int k);
	double compute_distence(Image img_test, Image img_train);
	double compute_distence(double* ct1, double* ct2, int size);
	friend double MATH_IMG::distL2(double* ct1, double* ct2, int size);
};

#endif