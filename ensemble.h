#pragma once

#include <vector>
#include "regression_tree.h"

//template <typename T>
class ensemble {
public:
    typedef regression_tree tree_type;
    typedef std::shared_ptr<tree_type> tree_ptr;

    explicit ensemble(size_t M, double learning_rate = 1.0) : M_(M), learning_rate_(learning_rate) {}
    // f - output file
    void serialize(const char*) {}
    // f - input file
    void deserialize(const char*) {}
    //void add_tree(const tree_ptr& tree) {}
    void learn(const feature_set& fs) {
        size_t n = fs.get_feature_count(); // number of features and index of label
        samples s(fs.get_samples()); // copy samples
        size_t N = s.size(); // number of samples

        // iteration 1
        size_t m = 1;
        double F0 = mean(s.begin(), s.end(), n);
        for (size_t i = 0; i < N; ++i) {
            double label = s[i][n];
            double residual = label - F0;
            s[i].push_back(residual);
        }
        print_samples(std::cout, s);
        tree_ptr t0(learn_tree(s, n, n + 1));
        //t0->print(std::cout);
        trees_.push_back(t0);

        // iterations 2+
        for (++m; m <= M_; ++m) {
            for (size_t i = 0; i < N; ++i) {
                double label = s[i][n];
                double residual = label - predict(s[i], n);
                s[i].push_back(residual);
            }
            print_samples(std::cout, s);
            tree_ptr t1(learn_tree(s, n, n + m));
            //t1->print(std::cout);
            trees_.push_back(t1);
        }
    }
    // x - features
    // n - feature count
    double predict(const std::vector<double>& x, size_t n) {
        double Fx = 0.0;
        for (size_t i = 0; i < trees_.size(); ++i)
            Fx += trees_[i]->predict(x, n);
        return Fx;
    }
    void print(std::ostream& os) const {
        os << "ENSEBMLE:\n";
        for (size_t i = 0; i < trees_.size(); ++i) {
            const tree_ptr& t = trees_[i];
            t->print(os);
        }
        os << "END OF ENSEMBLE" << std::endl;
    }

private:
    size_t M_;
    double learning_rate_;
    std::vector<tree_ptr> trees_;
};
