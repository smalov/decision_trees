#pragma once

#include <ostream>
#include "training_set.h"

// try different split criteria: gini, chi-square, entropy, etc
// todo: implement https://en.wikipedia.org/wiki/AdaBoost
class classification_tree {
public:
	classification_tree() : feature_(0), val_(0.0), lte_(0.0), gt_(0.0), alpha_(0.0) {}
	// w - index of column with weights
	// l - index of column with labels
	void learn(training_set& ts, size_t w, size_t l, std::ostream* logger = NULL) {
		size_t i = 0, j = 0;
		if (!split<binary_entropy>(ts, l, i, j))
			throw std::exception(); // return false;

		ts.sort(j);
		if (logger)
			print_split(*logger, ts, l, i, j);

		double val = ts.x(i - 1)[j];
		const double** it = ts.begin() + i;
		double lte = leaf_value(ts.begin(), it, l);
		double gt = leaf_value(it, ts.end(), l);

		double error = 0.0;
		for (size_t i = 0; i < ts.size(); ++i) {
			const double* x = ts.x(i);
			double wi = x[w]; // weight
			double yi = x[l]; // label
			double hi = x[j] <= val ? lte : gt;
			if (yi != hi)
				error += wi;
		}

		feature_ = j;
		val_ = val;
		lte_ = lte;
		gt_ = gt;
		alpha_ = 0.5 * log((1.0 - error) / error);
	}
	double predict(const double* x, size_t n) const {
		if (feature_ >= n)
			throw std::exception();
		return alpha_ * (x[feature_] <= val_ ? lte_ : gt_);
	}
	void print(std::ostream& os) const {
		os << "weak learner:"
			<< "\n\tfeature: " << feature_
			<< "\n\tval: " << val_
			<< "\n\tlte: " << lte_
			<< "\n\tgt: " << gt_
			<< "\n\talpha:" << alpha_
			<< std::endl;
	}
private:
	double leaf_value(const double** first, const double** last, size_t l) {
		size_t neg = count(first, last, l, -1.0), pos = count(first, last, l, 1.0);
		return (neg > pos ? -1.0 : 1.0);
	}

private:
	size_t feature_;
	double val_; // the top split value
	double lte_; // lte child split
	double gt_; // gt child split
	double alpha_;
};
