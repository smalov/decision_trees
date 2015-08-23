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

inline size_t count(const double** first, const double** last, size_t index, double value) {
	size_t c = 0;
	for (const double** it = first; it != last; ++it)
		if ((*it)[index] == value) c += 1;
	return c;
}

struct squared_error {
	double operator()(const double** first, const double** last, size_t index) const {
		double mval = mean(first, last, index);
		return (*this)(first, last, index, mval);
	}
	// no division by number of samples
	// sum((y - F)^2)
	double operator()(const double** first, const double** last, size_t index, double mean) const {
		double err = 0.0;
		for (const double** it = first; it != last; ++it) {
			double diff = (*it)[index] - mean;
			err += diff * diff;
		}
		return err;
	}
};

inline double mean_squared_error(const double** first, const double** last, size_t index) {
	squared_error f;
	double mval = mean(first, last, index);
	double err = f(first, last, index, mval);
	return err / std::distance(first, last);
}

// labels must have values -1 or +1 only
struct binary_entropy {
	double operator()(const double** first, const double** last, size_t index) const {
		size_t n = last - first;
		size_t c = count(first, last, index, -1.0);
		if (count(first, last, index, 1.0) != n - c)
			throw std::exception("invalid label value");
		double p1 = (double)c / n, p2 = (double)(n - c) / n;
		return -p1 * log2(p1) - p2 * log2(p2);
	}
};

inline double exponential_loss(const double** first, const double** last, size_t index) {
	return 0.0;
}
