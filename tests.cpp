#include <cassert>
#include "tests.h"
#include "feature_set.h"
#include "regression_tree.h"
#include "classification_tree.h"
#include "boosting.h"
#include "ensemble.h"
#include "decision_stump.h"

bool equal(double d1, double d2) {
	return abs(d1 - d2) < 0.001;
}

void run_tests(std::ostream& os) {
	//test_learning_of_regression_tree(os);
	//test_learning_of_stump_ensemble(os);
	test_learning_of_classification_tree(os);
}

void test_learning_of_classification_tree(std::ostream& os) {
	feature_set fs;
	load_feature_set_for_classification(fs, "training_set.txt");
	fs.print(os);

	ensemble<classification_tree, adaptive_boosting> e(5);
	e.learn_classifier(fs, &os);
	e.print(os);

	//training_set ts(fs);
	//classification_tree t;
	//t.learn(ts, ts.label_index(), &os);
}

void test_learning_of_regression_tree(std::ostream& os) {
	feature_set fs;
	load_feature_set(fs, "training_set.txt");
	training_set ts(fs);
	regression_tree t;
	t.learn(ts, ts.label_index(), &os);

	for (size_t i = 0; i < ts.size(); ++i)
		assert(t.predict(ts.x(i), ts.feature_count()) == ts.y(i)); 
}

void test_learning_of_stump_ensemble(std::ostream& os) {
	feature_set fs;
	load_feature_set(fs, "training_set.txt");
	fs.print(os);

	// learning
	ensemble<decision_stump, gradient_boosting> e(5);
	e.learn(fs, &os);
	e.print(os);

	assert(e.size() == 5);
	const decision_stump& t0 = e.tree(0);
	assert(t0.feature() == 1 && t0.val() == 1 && equal(t0.lte(), 0.522) && equal(t0.gt(), -0.145));
	const decision_stump& t1 = e.tree(1);
	assert(t1.feature() == 0 && t1.val() == 1 && equal(t1.lte(), 0.812) && equal(t1.gt(), 0.386));
	const decision_stump& t2 = e.tree(2);
	assert(t2.feature() == 1 && t2.val() == 7 && equal(t2.lte(), 0.0574) && equal(t2.gt(), -0.383));
	const decision_stump& t3 = e.tree(3);
	assert(t3.feature() == 1 && t3.val() == 4 && equal(t3.lte(), -0.147) && equal(t3.gt(), 0.16));
	const decision_stump& t4 = e.tree(4);
	assert(t4.feature() == 0 && t4.val() == 3 && equal(t4.lte(), 0.0911) && equal(t4.gt(), -0.0993));

	// evaluation -> precision/recall for a validation set 
	//std::unique_ptr<double[]> x(new double[n]);
	//std::cout << e.predict(x.get(), n);
}