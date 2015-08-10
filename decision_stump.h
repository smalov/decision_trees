#pragma once

#include <vector>
#include <ostream>
#include "feature_set.h"
#include "math.h"
#include "tree_learning.h"

// also known as one-level tree, weak learner, etc
class decision_stump {
public:
	decision_stump() : feature_(0), split_(0.0), lte_(0.0), gt_(0.0) {}
	decision_stump(size_t feature, double split_val, double lte_val, double gt_val)
		: feature_(feature), split_(split_val), lte_(lte_val), gt_(gt_val)
	{}
	double predict(const double* x, size_t n) {
		if (feature_ >= n)
			throw std::exception();
		return x[feature_] <= split_ ? lte_ : gt_;
	}
	// l - label index
	void learn(training_set& ts, size_t l, std::ostream* logger = NULL) {
		size_t i = 0; // feature vector index
		size_t j = 0; // feature index
		if (!split(ts, l, i, j))
			throw std::exception(); // return false;

		ts.sort(j);
		if (logger)
			print_split(*logger, ts, l, i, j);

		feature_ = j;
		split_ = ts.x(i - 1)[j];
		const double** middle = ts.begin() + i;
		lte_ = mean(ts.begin(), middle, l);
		gt_ = mean(middle, ts.end(), l);
		// return true;
	}
	void print(std::ostream& os) {
		os << "decision stump:"
			<< "\n\tfeature: " << feature_
			<< "\n\tsplit: " << split_
			<< "\n\tlte: " << lte_
			<< "\n\tgt: " << gt_
			<< std::endl;
	}
	size_t feature() const { return feature_; }
	double val() const { return split_; }
	double lte() const { return lte_; }
	double gt() const { return gt_; }
private:
	size_t feature_;
	double split_; // the top split
	double lte_; // lte child split
	double gt_; // gt child split
};
