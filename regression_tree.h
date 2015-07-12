#pragma once

#include <algorithm>
#include <vector>
#include <memory>
#include <limits>
#include <iomanip>
#include "feature_set.h"

class regression_tree {
public:
    //regression_tree(size_t num_leaves = 2) : num_leaves_(num_leaves) {}
    regression_tree()
        : feature_index_(0), split_value_(0.0), lte_value_(0.0), gt_value_(0.0) {}
    regression_tree(size_t feature_index, double split_value, double lte_value, double gt_value)
        : feature_index_(feature_index), split_value_(split_value), lte_value_(lte_value), gt_value_(gt_value)
    {
    }
    double predict(const std::vector<double>& x, size_t n) {
        if (feature_index_ >= n)
            throw std::exception();
        return x[feature_index_] <= split_value_ ? lte_value_ : gt_value_;
    }
    void print(std::ostream& os) {
        std::cout << "regression tree:\n\tfeature: " << feature_index_ << "\n\tsplit: "
            << split_value_ << "\n\tlte: " << lte_value_ << "\n\tgt: " << gt_value_ << std::endl;
    }
private:
    //size_t num_leaves_; // not used
    size_t feature_index_;
    double split_value_; // the top split
    double lte_value_; // lte child split
    double gt_value_; // gt child split
};

typedef std::shared_ptr<regression_tree> regression_tree_ptr;

struct sample_less {
    size_t i_; // index of feature
    sample_less(size_t i) : i_(i) {}
    bool operator()(const sample& left, const sample& right) const {
        return left[i_] < right[i_];
    }
};

void print_split(std::ostream& os, const samples& values, size_t label, size_t split_feature, size_t split_sample) {
    std::cout << "SPLIT(" << split_feature << "):\n";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i == split_sample)
            std::cout << "---------------\n";
        std::cout << values[i][split_feature] << "\t" << std::setprecision(3) << values[i][label] << "\n";
    }
    std::cout << std::endl;
}

double sum(samples::const_iterator first, samples::const_iterator last, size_t i) {
    double sum = 0.0;
    for (samples::const_iterator it = first; it != last; ++it)
        sum += (*it)[i];
    return sum;
}

double mean(samples::const_iterator first, samples::const_iterator last, size_t i) {
    return sum(first, last, i) / std::distance(first, last);
}

// no division by number of samples
// sum((y - F)^2)
double squared_error(samples::const_iterator first, samples::const_iterator last, double mean, size_t label_index) {
    double err = 0.0;
    for (samples::const_iterator it = first; it != last; ++it) {
        double diff = (*it)[label_index] - mean;
        err += diff * diff;
    }
    return err;
}

double mean_squared_error(samples::const_iterator first, samples::const_iterator middle, samples::const_iterator last, size_t i) {
    double mean1 = mean(first, middle, i), mean2 = mean(middle, last, i);
    double err1 = squared_error(first, middle, mean1, i), err2 = squared_error(middle, last, mean2, i);
    return (err1 + err2) / std::distance(first, last);
}

struct split_result {
    double value_;
    size_t feature_index_; // index of feature
    size_t sample_index_; // index of sample 
    split_result(double value, size_t feature_index, size_t sample_index)
        : value_(value), feature_index_(feature_index), sample_index_(sample_index) {}
};

// n - feature count
// l - index of label
// returns pair of index of feature and index of sample
split_result split(samples::iterator first, samples::iterator last, size_t n, size_t l) {
    if (first == last)
        throw std::exception();
    split_result res(0.0, 0, 0);
    double min_err = std::numeric_limits<double>::max();
    for (size_t i = 0; i < n; ++i) {
        std::sort(first, last, sample_less(i));
        samples::iterator it = first;
        double prev = (*it)[i];
        ++it;
        while (true) {
            while (it != last && (*it)[i] == prev)
                ++it;
            double err = mean_squared_error(first, it, last, l);
            if (min_err > err) {
                min_err = err;
                size_t j = it - first;
                res = split_result(prev, i, j);
            }
            if (it == last)
                break;
            prev = (*it)[i];
            ++it;
        }
    }
    return res;
}

//void assign_diff(samples::iterator first, samples::iterator last, size_t n) {
//    double m = mean(first, last, n);
//    size_t k = n + 1;
//    for (samples::iterator it = first; it != last; ++it)
//        (*it)[k] = (*it)[n] - m;
//}

//template <typename T>
//T* learn_tree(feature_set& features, size_t n, size_t l);

// n - number of features and index of label
// i - index of label (The pseudo-response in line 3 of Algorithm 1)
// samples is sorted after returning from the function
regression_tree* learn_tree(samples& s, size_t n, size_t l) {
    split_result res = split(s.begin(), s.end(), n, l);
    std::sort(s.begin(), s.end(), sample_less(res.feature_index_));
    print_split(std::cout, s, l, res.feature_index_, res.sample_index_);

    if (res.sample_index_ == 0)
        throw std::exception();

    double split_value = s[res.sample_index_ - 1][res.feature_index_];
    samples::iterator middle = s.begin() + res.sample_index_;
    double lte_value = mean(s.begin(), middle, l);
    double gt_value = mean(middle, s.end(), l);

    return new regression_tree(res.feature_index_, split_value, lte_value, gt_value);
}
