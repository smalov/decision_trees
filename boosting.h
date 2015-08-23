#pragma once

class gradient_boosting {
public:
	gradient_boosting() {}
	double initial_value(training_set& ts) const {
		return mean(ts.begin(), ts.end(), ts.label_index()); // mean value for all labels
	}
	double gradient(double label, double prediction) const {
		return label - prediction;
	}
};

// least_squares_regression
// least_absolute_deviation_regression 
// huber_loss_function 
// two_class_logistic_regression
// multi_class_logistic_regression

class adaptive_boosting {
public:
	adaptive_boosting() {}
	double initial_value(training_set& ts) const {
		return 1.0 / ts.size();
	}
	double gradient(double label, double weight) {
		return 0.0;
	}
};