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
	void print(std::ostream& os) {
		os << "decision stump:"
			<< "\n\tfeature: " << feature_index_
			<< "\n\tsplit: " << split_value_
			<< "\n\tlte: " << lte_value_
			<< "\n\tgt: " << gt_value_ << std::endl;
	}
	void learn(training_set& ts) {
		// i - feature vector index, j - feature index
		size_t i = 0, j = 0;
		if (!split(ts, i, j))
			throw std::exception(); // return false?

		ts.sort(j);
		//print_split(std::cout, fs.feature_data(), l, res.feature_index_, res.sample_index_);

		size_t l = ts.size(); // label index
		double split_value = ts.x(i - 1)[j];
		const double** middle = ts.begin() + i;
		double lte_value = mean(ts.begin(), middle, l);
		double gt_value = mean(middle, ts.end(), l);

		feature_index_ = j;
		split_value_ = split_value;
		lte_value_ = lte_value;
		gt_value_ = gt_value;
	}
private:
	size_t feature_index_;
	double split_value_; // the top split
	double lte_value_; // lte child split
	double gt_value_; // gt child split
};
