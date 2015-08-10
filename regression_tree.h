#pragma once

#include <algorithm>
#include <vector>
#include <memory>
#include <limits>
#include <iomanip>
#include "feature_set.h"
#include "math.h"
#include "tree_learning.h"

// todo:
// - different splitting criteria 
// - use stopping criteria
// - implement pruning 
// - add parameters/members:
//     num_leaves - maximum number of leaves in tree
//     min_docs_in_leaf - minimum number of documents in leaf
//     weight - ...
class regression_tree {
	struct node {
		int val_;
		int lte_;
		int gt_;
	};
public:
    regression_tree() {}
	double predict(const double* x, size_t n) { return 0.0; }
	void learn(training_set& ts, size_t l, std::ostream* logger = NULL) {}
    void print(std::ostream& os) {}
private:
    //size_t num_leaves_; 
	//size_t min_docs_in_leaf_;
	std::vector<node> nodes_;
};

typedef std::shared_ptr<regression_tree> regression_tree_ptr;
