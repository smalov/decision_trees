#pragma once

#include <vector>
#include <ostream>
#include "feature_set.h"
#include "math.h"
#include "tree_learning.h"

// also known as one-level tree, weak learner, etc
class decision_stump {
public:
	decision_stump() : feature_index_(0), split_value_(0.0), lte_value_(0.0), gt_value_(0.0) {}
	decision_stump(size_t feature_index, double split_value, double lte_value, double gt_value)
		: feature_index_(feature_index), split_value_(split_value), lte_value_(lte_value), gt_value_(gt_value)
	{}
	double predict(const double* x, size_t n) {
		if (feature_index_ >= n)
			throw std::exception();
		return x[feature_index_] <= split_value_ ? lte_value_ : gt_value_;
	}
	// l - label index
	void learn(training_set& ts, size_t l) {
		size_t i = 0; // feature vector index
		size_t j = 0; // feature index
		if (!split(ts, l, i, j))
			throw std::exception(); // return false;

		ts.sort(j);
		print_split(std::cout, ts, l, i, j);

		const double** middle = ts.begin() + i;
		feature_index_ = j;
		split_value_ = ts.x(i - 1)[j];
		lte_value_ = mean(ts.begin(), middle, l);
		gt_value_ = mean(middle, ts.end(), l);
		// return true;
	}
	void print(std::ostream& os) {
		os << "decision stump:"
			<< "\n\tfeature: " << feature_index_
			<< "\n\tsplit: " << split_value_
			<< "\n\tlte: " << lte_value_
			<< "\n\tgt: " << gt_value_ << std::endl;
	}
private:
	size_t feature_index_;
	double split_value_; // the top split
	double lte_value_; // lte child split
	double gt_value_; // gt child split
};
