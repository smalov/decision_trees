#pragma once

#include <vector>
#include <memory>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>

typedef std::vector<double*> feature_data;

void print_data(std::ostream& os, const feature_data& data, size_t n);
void delete_data(feature_data& data);

// feature_set -> feature_vector -> features, labels
// feature names
// bagging
// mixed numerical and categorical features 
class feature_set {
    size_t n_; // feature count and index of label
	feature_data data_;
public:
	feature_set() : n_(0) {}
	~feature_set() {
		delete_data(data_);
	}
	void initialize(feature_data& data, size_t n) {
        n_ = n - 1;
        data_.swap(data);
    }
	size_t size() const {
		return data_.size();
	}
	const double** begin() const {
		return (const double**)&data_[0];
	}
	const double** end() const {
		return begin() + data_.size();
	}
    void features() const {
        // not implemented
		// feature name, type, etc
    }
    // index of label is equal to feature count
    const size_t feature_count() const {
        return n_;
    }
	void copy(feature_data& other) const {
		// not implemented
	}
	void print(std::ostream& os) const {
		print_data(os, data_, n_);
	}
};

void load_feature_set(feature_set& fs, const char* file_name);
