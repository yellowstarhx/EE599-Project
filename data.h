// data.h
#ifndef DATA_H
#define DATA_H

#include <sstream>
#include <iostream>
#include <fstream>
#include <cassert>
#include <vector>
#include <string>
using namespace std;

class Image {
	int label;
	double* content;
	int size;
public:
	Image();
	Image(vector<string> row);
	void setImage(vector<string> row);
	void printImage();
	int get_size();
	double* get_content();
	int get_label();
	~Image();
};

class Dataset{
	int size;
	Image* dataset;
	string type;
public:
	Dataset();
	Dataset(string filename, string type);
	~Dataset();
	bool isTrain();
	void printDataset(); 
	void show1Image(int i);
	int get_size();
	Image* get_dataset();
};

#endif