#pragma once

#include <vector>


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

class squared_error {
	double val_;
public:
	squared_error(const double** first, const double** last, size_t index, size_t = 0)
		: val_(impl(first, last, index)) {}
	double gain(const double** first, const double** split, const double** last, size_t index, size_t = 0) const {
		double e = impl(first, split, index) + impl(split, last, index);
		return val_ > e ? val_ - e : 0.0;
	}
	double value() const { return val_; }
private:
	double impl(const double** first, const double** last, size_t index) const {
		double mval = mean(first, last, index);
		return impl(first, last, index, mval);
	}
	// no division by number of samples
	// sum((y - F)^2)
	double impl(const double** first, const double** last, size_t index, double mean) const {
		double err = 0.0;
		for (const double** it = first; it != last; ++it) {
			double diff = (*it)[index] - mean;
			err += diff * diff;
		}
		return err;
	}
};

inline double mean_squared_error(const double** first, const double** last, size_t index) {
	squared_error f(first, last, index);
	return f.value() / std::distance(first, last);
}

// labels must have values -1 or +1 only
class binary_entropy {
	size_t n_;
	double val_;
public:
	binary_entropy(const double** first, const double** last, size_t label_index, size_t weight_index)
		: n_(last - first), val_(impl(first, last, label_index, weight_index))
	{}
	// information gain
	double gain(const double** first, const double** split, const double** last, size_t label_index, size_t weight_index) const {
		double e1 = impl(first, split, label_index, weight_index) * (split - first) / n_;
		double e2 = impl(split, last, label_index, weight_index) * (last - split) / n_;
		return val_ - e1 - e2;
	}
	double value() const { return val_; }
private:
	double impl(const double** first, const double** last, size_t label_index, size_t weight_index) const {
		//size_t n = last - first;
		//size_t c = count(first, last, label_index, -1.0);
		//if (count(first, last, label_index, 1.0) != n - c)
		//	throw std::exception("invalid label value");
		double p1 = 0.0, p2 = 0.0;
		for (const double** it = first; it != last; ++it) {
			const double y = (*it)[label_index];
			if (y == -1.0)
				p1 += (*it)[weight_index];
			else if (y == 1.0)
				p2 += (*it)[weight_index];
			else
				throw std::exception("unexpected label value");
		}
		double e = 0.0;
		if (p1 > 0.0) e -= p1 * log2(p1);
		if (p2 > 0.0) e -= p2 * log2(p2);
		return e;
	}
};
