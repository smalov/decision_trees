#pragma once

#include <vector>

// entropy
// least squares
// etc

double sum(const double** first, const double** last, size_t index) {
	double sum = 0.0;
	for (const double** it = first; it != last; ++it)
		sum += (*it)[index];
	return sum;
}

double mean(const double** first, const double** last, size_t index) {
	return sum(first, last, index) / std::distance(first, last);
}

// no division by number of samples
// sum((y - F)^2)
double squared_error(const double** first, const double** last, size_t label, double mean) {
	double err = 0.0;
	for (const double** it = first; it != last; ++it) {
		double diff = (*it)[label] - mean;
		err += diff * diff;
	}
	return err;
}

double mean_squared_error(const double** first, const double** middle, const double** last, size_t index) {
	double mean1 = mean(first, middle, index), mean2 = mean(middle, last, index);
	double err1 = squared_error(first, middle, index, mean1), err2 = squared_error(middle, last, index, mean2);
	return (err1 + err2) / std::distance(first, last);
}
