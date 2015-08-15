#pragma once

#include <algorithm>
#include "training_set.h"
#include "math.h"

inline bool split(training_set& ts, size_t i1, size_t i2, size_t l, size_t& i, size_t& j) {
	const double e_max = squared_error(ts.begin() + i1, ts.begin() + i2, l);
	double e_min = e_max;// DBL_MAX;
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
			double e = squared_error(first, it, l) + squared_error(it, last, l);
			// note: mean_squared_error() does not work here
			if (e_min > e) {
				e_min = e;
				i = it - ts.begin();// first;
				j = k;
			}
			if (it == last)
				break;
			prev = (*it)[k];
			++it;
		}
	}
	return e_min < e_max;
}

// i - index of feature vector
// j - index of feature
// returns pair of index of feature and index of sample
inline bool split(training_set& ts, size_t l, size_t& i, size_t& j) {
	return split(ts, 0, ts.size(), l, i, j);
}

// training set must be sorted by j-th feature
inline void print_split(std::ostream& os, const training_set& ts, size_t i1, size_t i2, size_t l, size_t i, size_t j) {
	os << "SPLIT(x" << j << ")\n";
	for (size_t f = 0; f < ts.feature_count(); ++f)
		os << "x" << f << "\t";
	os << "y\tgrad\n----------------------------\n";
	for (size_t k = i1; k < i2; ++k) {
		if (k == i)
			os << "----------------------------\n";
		for (size_t f = 0; f < ts.feature_count(); ++f)
			os << ts.x(k)[f] << "\t";
		os << ts.y(k) << "\t" << ts.gradient(k) << "\n";
	}
	os << "END OF SPLIT" << std::endl;
}

inline void print_split(std::ostream& os, const training_set& ts, size_t l, size_t i, size_t j) {
	print_split(os, ts, 0, ts.size(), l, i, j);
}
