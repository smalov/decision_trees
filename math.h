#pragma once

#include <vector>

// entropy
// least squares
// etc

inline double sum(const double** first, const double** last, size_t index) {
	double sum = 0.0;
	for (const double** it = first; it != last; ++it)
		sum += (*it)[index];
	return sum;
}

inline double mean(const double** first, const double** last, size_t index) {
	return sum(first, last, index) / std::distance(first, last);
}

// no division by number of samples
// sum((y - F)^2)
inline double squared_error(const double** first, const double** last, size_t label, double mean) {
	double err = 0.0;
	for (const double** it = first; it != last; ++it) {
		double diff = (*it)[label] - mean;
		err += diff * diff;
	}
	return err;
}

inline double squared_error(const double** first, const double** last, size_t index) {
	double mval = mean(first, last, index);
	return squared_error(first, last, index, mval);
}

inline double mean_squared_error(const double** first, const double** last, size_t index) {
	double mval = mean(first, last, index);
	double err = squared_error(first, last, index, mval);
	return err / std::distance(first, last);
}
