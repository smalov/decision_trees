#pragma once

#include <algorithm>
#include "training_set.h"
#include "math.h"

// i - index of feature vector
// j - index of feature
// returns pair of index of feature and index of sample
inline bool split(training_set& ts, size_t l, size_t& i, size_t& j) {
	double e_min = DBL_MAX;
	for (size_t k = 0; k < ts.feature_count(); ++k) {
		ts.sort(k);
		const double** first = ts.begin();
		const double** last = ts.end();
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

inline void print_split(std::ostream& os, const training_set& ts, size_t l, size_t i, size_t j) {
	os << "SPLIT(" << j << ")\n";
	for (size_t k = 0; k < ts.size(); ++k) {
		if (k == i)
			os << "----------------------------\n";
		os << ts.x(k)[j] << "\t" << ts.y(k) << "\t" << ts.gradient(k) << "\n";
	}
}
