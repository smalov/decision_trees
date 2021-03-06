#include <iostream>
#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include "decision_stump.h"
#include "regression_tree.h"
#include "feature_set.h"
#include "ensemble.h"
#include "boosting.h"
#include "tests.h"


// parameters: command (learn/predict), training set, model
// store/load learned tree/ensemble
// learing to rank
int main(int argc, char* argv[]) {
	if (argc > 2 && strcmp(argv[1], "run_tests") == 0) {
		run_tests(std::cout);
		return 0;
	}

	std::ostream& os = std::cout;
	test_learning_of_classification_tree(os);
	test_learning_of_regression_tree(os);
	test_learning_of_stump_ensemble(os);

	std::cout << "OK" << std::endl;
	return 0;
}
