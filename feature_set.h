#pragma once

#include <vector>
#include <memory>
#include <iostream>
#include <fstream>
#include <string>
#include "sample.h"

class feature_set {
    size_t n_;
    samples data_;
public:
    feature_set(samples& s, size_t n) {
        n_ = n;
        data_.swap(s);
    }
    void print(std::ostream& os) const {
        print_samples(os, data_);
    }
    const samples& get_samples() const {
        return data_;
    }
    void get_feature_names() const {
        // not implemented
    }
    // index of label is equal to feature count
    const size_t get_feature_count() const {
        return n_;
    }
};

typedef std::shared_ptr<feature_set> feature_set_ptr;

feature_set_ptr load_features(const char* file_name)
{
	std::ifstream f(file_name);
	std::string line;
    // read header
    if (!std::getline(f, line))
        throw std::exception("empty file");
    size_t n = 0; // number of columns
    size_t first = 0, last = 0;
    for (; last != std::string::npos;) {
        last = line.find('\t', first);
        first = last + 1;
        ++n;
    }
    // read values
    samples s;
	while (std::getline(f, line)) {
        s.push_back(sample(n));
        first = 0, last = 0;
        size_t i = 0;
        while (i < n) {
			last = line.find('\t', first);
            double val = strtod(line.c_str() + first, nullptr);
            s.back()[i++] = val;
            if (last == std::string::npos)
                break;
			first = last + 1;
		}
        if (i != n)
            throw std::exception("invalid number of values");
	}
	return feature_set_ptr(new feature_set(s, n - 1));
}
