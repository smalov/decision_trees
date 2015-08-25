#pragma once

#include <vector>
#include "training_set.h"
#include "math.h"

// namespace decision_tree_learning_framework {

template <typename Tree, typename Boosting>
class ensemble {
public:
    typedef Tree tree_type;
    typedef std::shared_ptr<tree_type> tree_ptr;
	typedef Boosting boosting_type;

	// iteration count - max number of trees
	// learing rate - not used, tbd
    explicit ensemble(size_t iteration_count, double learning_rate = 1.0)
		: iteration_count_(iteration_count), learning_rate_(learning_rate)
	{}
    // f - output file
    void serialize(const char*) {}
    // f - input file
    void deserialize(const char*) {}
    //void add_tree(const tree_ptr& tree) {}
	void learn(const feature_set& fs, std::ostream* logger = NULL) {
		training_set ts(fs);
		size_t n = ts.feature_count();
		boosting_type boosting;

		double F0 = boosting.initial_value(ts);
		for (size_t m = 0; m < iteration_count_; ++m) {
			for (size_t i = 0; i < ts.size(); ++i) {
				double label = ts.y(i);
				double prediction = (m == 0 ? F0 : predict(ts.x(i), n));
				double gradient = boosting.gradient(label, prediction);
				ts.set_gradient(i, gradient);
			}
			if (logger)
				ts.print(*logger);
			tree_ptr t(new tree_type());
			t->learn(ts, ts.gradient_index(), logger);
			if (logger)
				t->print(*logger);
			trees_.push_back(t);
		}
	}
	void learn_classifier(const feature_set& fs, std::ostream* logger = NULL) {
		training_set ts(fs);
		size_t n = ts.feature_count();
		boosting_type boosting;

		double w0 = boosting.initial_value(ts);
		for (size_t i = 0; i < ts.size(); ++i)
			ts.set_weight(i, w0);
		if (logger) ts.print(*logger);
		for (size_t m = 0; m < iteration_count_; ++m) {
			//for (size_t i = 0; i < ts.size(); ++i) {
			//	double label = ts.y(i);
			//	double prediction = (m == 0 ? F0 : predict(ts.x(i), n));
			//	double gradient = boosting.gradient(label, prediction);
			//	ts.set_gradient(i, gradient);
			//}
			//if (logger) ts.print(*logger);
			tree_ptr t(new tree_type());
			t->learn(ts, ts.weight_index(), ts.label_index(), logger);
			if (logger) t->print(*logger);
			trees_.push_back(t);
			double total_weight = 0.0;
			for (size_t i = 0; i < ts.size(); ++i) {
				double w = ts.w(i) * exp(-ts.y(i) * t->predict(ts.x(i), n));
				ts.set_weight(i, w);
				total_weight += w;
			}
			for (size_t i = 0; i < ts.size(); ++i)
				ts.set_weight(i, ts.w(i) / total_weight);
			if (logger) ts.print(*logger);
		}
	}
	// x - features
    // n - feature count
	double predict(const double* x, size_t n) const {
		double Fx = 0.0;
		for (size_t i = 0; i < trees_.size(); ++i)
			Fx += trees_[i]->predict(x, n);
		return Fx;
	}
	double predict_class(const double* x, size_t n) const {
		double Fx = 0.0;
		for (size_t i = 0; i < trees_.size(); ++i)
			Fx += trees_[i]->predict(x, n);
		return Fx > 0.0 ? 1.0 : -1.0;
	}
	void print(std::ostream& os) const {
        os << "ENSEBMLE:\n";
        for (size_t i = 0; i < trees_.size(); ++i)
            trees_[i]->print(os);
        os << "END OF ENSEMBLE" << std::endl;
    }
	size_t size() const { return trees_.size(); }
	const tree_type& tree(size_t i) { return *trees_[i]; }
private:
	size_t iteration_count_;
    double learning_rate_;
    std::vector<tree_ptr> trees_;
};
