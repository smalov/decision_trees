#include "tests.h"
#include "feature_set.h"
#include "regression_tree.h"
#include "boosting.h"
#include "ensemble.h"
#include "decision_stump.h"

void run_tests(std::ostream& os) {
	test_learning_of_regression_tree(os);
	test_learning_of_stump_ensemble(os);
}

void test_learning_of_regression_tree(std::ostream& os) {
	feature_set fs;
	load_feature_set(fs, "training_set.txt");
	training_set ts(fs);
	regression_tree t;
	t.learn(ts, ts.label_index(), &os);
}

void test_learning_of_stump_ensemble(std::ostream& os) {
	feature_set fs;
	load_feature_set(fs, "training_set.txt");
	fs.print(os);

	// learning
	ensemble<decision_stump, gradient_boosting> e(5);
	e.learn(fs, &os);
	e.print(os);

	// evaluation -> precision/recall for a validation set 
	//std::unique_ptr<double[]> x(new double[n]);
	//std::cout << e.predict(x.get(), n);
}