#include "KNN.h"

KNN::KNN(){
	k = 0;
	train_data = NULL;
	test_data = NULL;
}

KNN::~KNN() {
	delete train_data, test_data;
}

void KNN::set_k(int n){
	k = n;
}

void KNN::set_train_data(string filename) {
	train_data = new Dataset(filename, "train");
}

void KNN::set_test_data(string filename) {
	test_data = new Dataset(filename, "test");
}

int KNN::predict(Image img) {
	Image* train_imgs = train_data -> get_dataset();
	int dataset_size = train_data -> get_size();
	// k
	double cur_max = 0;
	int where_max = 0;	// index of current max distance in the list
	vector<pair<int, double>> candicates;
	candicates.clear();
	for (int i = 0; i < dataset_size; ++i) {
		cout << endl << i << " -th image in train set(label: " << train_imgs[i].get_label() << ")" << endl;
		// double cur_distance = compute_distence(img, train_imgs[i]);	// Error
		double cur_distance = compute_distence(img.get_content(), train_imgs[i].get_content(), img.get_size());
		cout << "*******distance: " << cur_distance << endl;
		// *if the distance has been already greater than cur_max, we don't need to preceed
		if (candicates.size() < k) {
			candicates.push_back(make_pair(train_imgs[i].get_label(), cur_distance));
		} else if (cur_distance < cur_max) {
			candicates[where_max] = make_pair(train_imgs[i].get_label(), cur_distance);
		}
		// *use a heap
		cur_max = 0;
		where_max = 0;
		for (int j = 0; j < candicates.size(); ++j) {
			if (candicates[j].second > cur_max) {
				cur_max = candicates[j].second;
				where_max = j;
			}
			cout << candicates[j].first << "," << candicates[j].second << " ";
		}
		cout << endl;
		assert(candicates.size() <= k);
	}
	// vote
	map<int, int> vote;
	cout << "Counting votes:";
	for (int i = 0; i < k; ++i) {
		cout << candicates[i].first << "," << candicates[i].second << " ";
		if (vote.find(candicates[i].first) != vote.end()) {
			vote[candicates[i].first] += 1;
		} else {
			vote[candicates[i].first] = 1;
		}
	}
	int result = 0;
	int majority = 0;
	cout << endl << "Voting:";
	for (auto itr = vote.begin(); itr != vote.end(); ++itr) {
    	if (itr->second > majority) {
    		majority = itr->second;
    		result = itr->first;
    	}
    	cout << itr->first << ":" << itr->second << " ";
  	}
  	cout << endl;
  	assert(result != 0);
	// return
	return result;
}

int KNN::predict(Image img, int j) {
	set_k(j);
	predict(img);
}

void KNN::predict_all() {
	cout << "Excuting KNN, with K = " << k << endl;
	Image* test_imgs = test_data -> get_dataset();
	int dataset_size = test_data -> get_size();
	cout << "Importing test set, the test set has " << dataset_size << " images." << endl;
	cout << "Train set size is " <<  train_data->get_size() << endl;
	int correct_count = 0;
	for (int i = 0; i < dataset_size; ++i) {
	// for (int i = 0; i < 1; ++i) {
		cout << "Computing the " << i << " -th test image...";
		int result = predict(test_imgs[i]);
		int answer = test_imgs[i].get_label();
		cout << "predict: " << result << " In fact: " << answer;
		if (result == answer) {
			correct_count++;
			cout << "---> Correct!" << endl;
		} else {
			cout << "---> Wrong." << endl;
		}
	}
	cout << "correct_count: " << correct_count;
	double accuracy = (double)correct_count / dataset_size;
	cout << endl << "The overall accuracy is " << accuracy << endl;
}

void KNN::predict_all(int j) {
	set_k(j);
	predict_all();
}

double KNN::compute_distence(Image img_test, Image img_train) {
	int n = img_test.get_size();
	assert(n == img_train.get_size());
	double distance = 0;
	for (int i = 0; i < n; i++) {
		distance += pow((img_test.get_content()[i] - img_train.get_content()[i]), 2);
	}
	distance = sqrt(distance);
	cout << "In func compute(): " << distance << endl;
	return distance;
}

double KNN::compute_distence(double* ct1, double* ct2, int size) {
	double distance = 0;
	for (int i = 0; i < size; i++) {
		distance += pow((ct1[i] - ct2[i]), 2);
	}
	assert(distance < size);
	// distance = sqrt(distance);
	// cout << "In func compute(): " << distance << endl;
	return distance;	
}