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
		size_t feature_;
		double val_;
		size_t lte_;
		size_t gt_;
		// node is leaf if lte_ == gt_ == 0
		node() : feature_(0), val_(0.0), lte_(0), gt_(0) {}
	};
public:
    regression_tree() {}
	double predict(const double* x, size_t n) { return 0.0; }
	void learn(training_set& ts, size_t l, std::ostream* logger = NULL) {
		split3(ts, 0, ts.size(), l); // root split: 0..N
	}
private:
	// recursive function
	// returns index of new node 
	size_t split3(training_set& ts, size_t i1, size_t i2, size_t l) {
		//if (i2 - i1 < min_docs_in_leaf) create leaf node

		size_t i = 0; // feature vector index 
		size_t j = 0; // feature index

		const size_t ni = nodes_.size(); // node index
		nodes_.push_back(node());
		node& n = nodes_.back();

		if (!split(ts, l, i, j)) {
			// create leaf node
			n.val_ = mean(ts.begin() + i1, ts.begin() + i2, l);
			return ni;
		}
		//else if ((i - i1) < min_docs_in_leaf || (i2 - i) < min_docs_in_leaf) {
		//    create leaf node
		//} else if one of the child splits will contain less docs than min_docs_in_leaf 
		//    this node must be leaf as well

		ts.sort(j);

		n.feature_ = j;
		n.val_ = ts.x(i - 1)[j];
		n.lte_ = split3(ts, i1, i, l);
		n.gt_ = split3(ts, i, i2, l);
		return ni;
	}
    void print(std::ostream& os) {}
private:
    //size_t num_leaves_; 
	//size_t min_docs_in_leaf_;
	std::vector<node> nodes_;
};

typedef std::shared_ptr<regression_tree> regression_tree_ptr;
