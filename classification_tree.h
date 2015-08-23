#pragma once

#include <ostream>
#include "training_set.h"

// try different split criteria: gini, chi-square, entropy, etc
// todo: implement https://en.wikipedia.org/wiki/AdaBoost
class classification_tree {
public:
	void learn(training_set& ts, size_t l, std::ostream* logger = NULL) {}
	double predict(const double* x, size_t n) const {
		return 0.0;
	}
	void print(std::ostream& os) const {}
};
