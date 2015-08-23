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
		bool is_leaf() const { return lte_ == gt_; }
	};
public:
    regression_tree() {}
	double predict(const double* x, size_t n) const {
		size_t i = root_;
		while (true) {
			const node& n = nodes_[i];
			if (n.is_leaf())
				return n.val_;
			i = (x[n.feature_] <= n.val_ ? n.lte_ : n.gt_);
		}
		return 0.0;
	}
	void learn(training_set& ts, size_t l, std::ostream* logger = NULL) {
		root_ = split3(ts, 0, ts.size(), l, logger); // root split: 0..N
		if (logger) print(*logger);
	}
    void print(std::ostream& os) const {
		os << "TREE\n";
		print(root_, 0, os);
		os << "END OF TREE" << std::endl;
	}
private:
	// recursive function
	// returns index of new node 
	size_t split3(training_set& ts, size_t i1, size_t i2, size_t l, std::ostream* logger = NULL) {
		//if (i2 - i1 < min_docs_in_leaf) create leaf node

		size_t i = 0, j = 0;
		if (!split(ts, i1, i2, l, i, j, squared_error())) {
			// create leaf node
			node n;
			n.val_ = mean(ts.begin() + i1, ts.begin() + i2, l);
			nodes_.push_back(n);
			return nodes_.size() - 1;
		}
		//else if ((i - i1) < min_docs_in_leaf || (i2 - i) < min_docs_in_leaf) {
		//    create leaf node
		//} else if one of the child splits will contain less docs than min_docs_in_leaf 
		//    this node must be leaf as well

		ts.sort(i1, i2, j);
		if (logger) print_split(*logger, ts, i1, i2, l, i, j);

		node n;
		n.feature_ = j;
		n.val_ = ts.x(i - 1)[j];
		n.lte_ = split3(ts, i1, i, l, logger);
		n.gt_ = split3(ts, i, i2, l, logger);
		nodes_.push_back(n);
		return nodes_.size() - 1;
	}
	void print(size_t i, size_t level, std::ostream& os) const {
		const node& n = nodes_[i];
		const std::string indent((level <= 1 ? 0 : level - 1) * 5, ' ');
		const std::string prefix(level == 0 ? "" : "  +--");
		if (n.is_leaf()) {
			os << indent << prefix << "[" << n.val_ << "]\n";
			return;
		}
		os << indent << prefix << "[x" << n.feature_ << " <= " << n.val_ << "]\n";
		print(n.lte_, level + 1, os);
		print(n.gt_, level + 1, os);
	}
private:
    //size_t num_leaves_; 
	//size_t min_docs_in_leaf_;
	std::vector<node> nodes_;
	size_t root_; // index of the root node
};

typedef std::shared_ptr<regression_tree> regression_tree_ptr;
