#pragma once 

#include "feature_set.h"

// feature_vector_less
struct x_less {
	size_t j_;
	x_less(size_t j) : j_(j) {}
	bool operator()(const double* x1, const double* x2) const {
		return x1[j_] < x2[j_];
	}
};

class training_set {
	feature_data data_;
	size_t n_; // feature count and index of label
public:
	training_set(const feature_set& fs) {
		copy_data(fs);
	}
	~training_set() {
		delete_data(data_);
	}
	size_t size() const {
		return data_.size();
	}
	size_t feature_count() const {
		return n_;
	}
	size_t label_index() const {
		return n_;
	}
	size_t gradient_index() const {
		return n_ + 1;
	}
	const double* feature_vector(size_t i) const {
		return data_[i];
	}
	const double* x(size_t i) const { // alias for feature_vector(i)
		return feature_vector(i);
	}
	double label(size_t i) const {
		return data_[i][n_];
	}
	double gradient(size_t i) const {
		return data_[i][n_ + 1];
	}
	double y(size_t i) const { // alias for label(i)
		return label(i);
	}
	void set_gradient(size_t i, double val) {
		data_[i][n_ + 1] = val;
	}
	void sort(size_t j) {
		sort(0, size(), j);
	}
	void sort(size_t i1, size_t i2, size_t j) {
		std::sort(data_.begin() + i1, data_.begin() + i2, x_less(j));
	}
	const double** begin() const {
		return (const double**)&data_[0];
	}
	const double** end() const {
		return begin() + data_.size();
	}
	void print(std::ostream& os) const {
		for (size_t i = 0; i < data_.size(); ++i) {
			const double* x = data_[i];
			for (size_t j = 0; j <= (n_ + 1); ++j)
				os << "\t" << data_[i][j];
			os << std::endl;
		}
	}
private:
	void copy_data(const feature_set& fs) {
		n_ = fs.feature_count();
		data_.reserve(fs.size());
		for (const double** it = fs.begin(); it != fs.end(); ++it) {
			double* p = new double[n_ + 2]; // + 1 element for gradient 
			memcpy(p, *it, (n_ + 1) * sizeof(double)); // copy x and y
			p[n_ + 1] = 0.0; // gradient
			data_.push_back(p);
		}
	}
};