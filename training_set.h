#pragma once 

#include "feature_set.h"

// feature_vector_less
struct x_less {
	size_t i_;
	x_less(size_t i) : i_(i) {}
	bool operator()(const double* x1, const double* x2) const {
		return x1[i_] < x2[i_];
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
	const double* feature_vector(size_t i) const {
		return data_[i];
	}
	const double* x(size_t i) const { // alias for feature_vector(i)
		return feature_vector(i);
	}
	double label(size_t i) const {
		return data_[i][n_];
	}
	double y(size_t i) const { // alias for label(i)
		return label(i);
	}
	void set_label(size_t i, double val) {
		data_[i][n_] = val;
	}
	void sort(size_t i) {
		sort(0, size(), i);
	}
	void sort(size_t b, size_t e, size_t i) {
		std::sort(data_.begin() + b, data_.begin() + e, x_less(i));
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
			for (size_t j = 0; j <= n_; ++j)
				os << "\t" << data_[i][j];
			os << std::endl;
		}
	}
private:
	void copy_data(const feature_set& fs) {
		n_ = fs.feature_count();
		data_.reserve(fs.size());
		for (const double** it = fs.begin(); it != fs.end(); ++it) {
			double* p = new double[n_ + 1];
			memcpy(p, *it, (n_ + 1) * sizeof(double));
			data_.push_back(p);
		}
	}
};