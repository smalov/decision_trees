#pragma once

#include <algorithm>
#include "training_set.h"
#include "math.h"

template <typename Tree, typename Pruning>
class tree_learner {
public:
	typedef Tree tree_type;
	typedef Pruning pruning_type;

	tree_learner() {}
	void learn(tree_type& t) {}
};

struct split_result {
	//double value_;
	size_t feature_index_; // index of feature
	size_t sample_index_; // index of sample 
	split_result(/*double value,*/ size_t feature_index, size_t sample_index)
		: /*value_(value),*/ feature_index_(feature_index), sample_index_(sample_index) {}
};

// i - index of feature vector
// j - index of feature
// returns pair of index of feature and index of sample
bool split(training_set& ts, size_t& i, size_t& j) {
	double e_min = std::numeric_limits<double>::max();
	size_t l = ts.label_index();
	for (size_t k = 0; k < ts.feature_count(); ++k) {
		ts.sort(k);
		const double** first = ts.begin();
		const double** last = ts.end();
		const double** it = first;
		double prev = (*it)[k];
		++it;
		while (true) {
			// skip equal values
			while (it != last && (*it)[k] == prev)
				++it;
			double e = mean_squared_error(first, it, last, l);
			if (e_min > e) {
				e_min = e;
				i = it - first;
				j = k;
			}
			if (it == last)
				break;
			prev = (*it)[k];
			++it;
		}
	}
	return true;
}

//template <typename Tree>
//void learn_tree(Tree& t, const feature_set& fs) {
//	split_result res = split(s.begin(), s.end(), n, l);
//	std::sort(s.begin(), s.end(), sample_less(res.feature_index_));
//	print_split(std::cout, s, l, res.feature_index_, res.sample_index_);
//
//	if (res.sample_index_ == 0)
//		throw std::exception();
//
//	double split_value = s[res.sample_index_ - 1][res.feature_index_];
//	samples::iterator middle = s.begin() + res.sample_index_;
//	double lte_value = mean(s.begin(), middle, l);
//	double gt_value = mean(middle, s.end(), l);
//
//	return new regression_tree(res.feature_index_, split_value, lte_value, gt_value);
//}

//template <typename Ensemble, typename Tree>
//void learn_ensemble(Ensemble& e, Tree& t) {
//
//}


