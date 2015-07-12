#pragma once

#include <vector>
#include <iostream>

typedef std::vector<double> sample;
typedef std::vector<sample> samples;

void print_samples(std::ostream& os, const samples& data) {
    std::cout << "SAMPLES:\n";
    for (size_t i = 0; i < data.size(); ++i) {
        const sample& s = data[i];
        os << std::setprecision(3) << s[0];
        for (size_t j = 1; j < s.size(); ++j)
            os << "\t" << std::setprecision(3) << s[j];
        os << "\n";
    }
    os << std::endl;
}
