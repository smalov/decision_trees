#pragma once

#include <vector>
#include <ostream>
#include "feature_set.h"
#include "math.h"
#include "tree_learning.h"

// also known as one-level tree, weak learner, etc
class decision_stump {
public:
	decision_stump() : feature_(0), val_(0.0), lte_(0.0), gt_(0.0) {}
	decision_stump(size_t feature, double val, double lte_val, double gt_val)
		: feature_(feature), val_(val), lte_(lte_val), gt_(gt_val)
	{}
	double predict(const double* x, size_t n) {
		if (feature_ >= n)
			throw std::exception();
		return x[feature_] <= val_ ? lte_ : gt_;
	}
	// l - label index
	void learn(training_set& ts, size_t l, std::ostream* logger = NULL) {
		size_t i = 0, j = 0; 
		if (!split<squared_error>(ts, l, i, j))
			throw std::exception(); // return false;

		ts.sort(j);
		if (logger) print_split(*logger, ts, l, i, j);

		feature_ = j;
		val_ = ts.x(i - 1)[j];
		const double** it = ts.begin() + i;
		lte_ = mean(ts.begin(), it, l);
		gt_ = mean(it, ts.end(), l);
		// return true;
	}
	void print(std::ostream& os) {
		os << "decision stump:"
			<< "\n\tfeature: " << feature_
			<< "\n\tval: " << val_
			<< "\n\tlte: " << lte_
			<< "\n\tgt: " << gt_
			<< std::endl;
	}
	size_t feature() const { return feature_; }
	double val() const { return val_; }
	double lte() const { return lte_; }
	double gt() const { return gt_; }
private:
	size_t feature_;
	double val_; // the top split value
	double lte_; // lte child split
	double gt_; // gt child split
};
