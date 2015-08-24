#pragma once

#include <algorithm>
#include "training_set.h"
#include "math.h"

template <typename F>
inline bool split(training_set& ts, size_t i1, size_t i2, size_t l, size_t& i, size_t& j) {
	const size_t w = ts.weight_index();
	const F f(ts.begin() + i1, ts.begin() + i2, l, w);
	double gain = 0.0;
	for (size_t k = 0; k < ts.feature_count(); ++k) {
		ts.sort(i1, i2, k);
		const double** const first = ts.begin() + i1;
		const double** const last = ts.begin() + i2;
		const double** it = first;
		double prev = (*it)[k];
		++it;
		while (true) {
			while (it != last && (*it)[k] == prev)
				++it; // skip equal values
			double g = f.gain(first, it, last, l, w);
			if (gain < g) {
				gain = g;
				i = it - ts.begin();
				j = k;
			}
			if (it == last)
				break;
			prev = (*it)[k];
			++it;
		}
	}
	return gain > 0.0;
}

// i - index of feature vector
// j - index of feature
// returns pair of index of feature and index of sample
template <typename F>
inline bool split(training_set& ts, size_t l, size_t& i, size_t& j) {
	return split<F>(ts, 0, ts.size(), l, i, j);
}

// training set must be sorted by j-th feature
inline void print_split(std::ostream& os, const training_set& ts, size_t i1, size_t i2, size_t l, size_t i, size_t j) {
	os << "SPLIT(x" << j << ")\n";
	for (size_t f = 0; f < ts.feature_count(); ++f)
		os << "x" << f << "\t";
	os << "y\tgrad\tweight\n--------------------------------------\n";
	for (size_t k = i1; k < i2; ++k) {
		if (k == i)
			os << "--------------------------------------\n";
		for (size_t f = 0; f < ts.feature_count(); ++f)
			os << ts.x(k)[f] << "\t";
		os << ts.label(k) << "\t" << ts.gradient(k) << "\t" << ts.weight(k) << "\n";
	}
	os << "END OF SPLIT" << std::endl;
}

inline void print_split(std::ostream& os, const training_set& ts, size_t l, size_t i, size_t j) {
	print_split(os, ts, 0, ts.size(), l, i, j);
}
