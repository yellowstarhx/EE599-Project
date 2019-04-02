// data.h
#ifndef DATA_H
#define DATA_H

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
	~Image();
};

class Dataset{
	int size;
	Image* dataset;
	string type;
public:
	Dataset(string filename, string type);
	~Dataset();
	bool isTrain();
	void printDataset();
	void show1Image(int i);
};

#endif