#pragma once

class gradient_boosting {
public:
	gradient_boosting() {}
	double gradient(double label, double prediction) {
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
	double gradient(double label, double prediction) {
		return 0.0;
	}
};