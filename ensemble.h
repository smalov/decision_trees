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

    explicit ensemble(size_t iteration_count, double learning_rate = 1.0)
		: iteration_count_(iteration_count), learning_rate_(learning_rate)
	{}
    // f - output file
    void serialize(const char*) {}
    // f - input file
    void deserialize(const char*) {}
    //void add_tree(const tree_ptr& tree) {}
    void learn(const feature_set& fs) {
        training_set ts(fs);
		size_t n = ts.feature_count();
		boosting_type boosting;

        double F0 = mean(ts.begin(), ts.end(), n); // mean value for all labels
        for (size_t m = 0; m < iteration_count_; ++m) {
            for (size_t i = 0; i < ts.size(); ++i) {
                double label = ts.y(i);
				double prediction = (m == 0 ? F0 : predict(ts.x(i), n));
				double gradient = boosting.gradient(label, prediction); 
				ts.set_gradient(i, gradient); 
            }
            ts.print(std::cout);
			tree_ptr t(new tree_type());
			t->learn(ts, ts.gradient_index());
			t->print(std::cout);
            trees_.push_back(t);
        }
    }
    // x - features
    // n - feature count
    double predict(const double* x, size_t n) {
        double Fx = 0.0;
		for (size_t i = 0; i < trees_.size(); ++i)
			Fx += trees_[i]->predict(x, n);
        return Fx;
    }
    void print(std::ostream& os) const {
        os << "ENSEBMLE:\n";
        for (size_t i = 0; i < trees_.size(); ++i)
            trees_[i]->print(os);
        os << "END OF ENSEMBLE" << std::endl;
    }

private:
	size_t iteration_count_;
    double learning_rate_;
    std::vector<tree_ptr> trees_;
};
