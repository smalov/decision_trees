#include <iostream>
#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include "decision_stump.h"
#include "feature_set.h"
#include "ensemble.h"
#include "boosting.h"

void run_tests(std::ostream&);

// learing to rank
// ...
int main(int argc, char* argv[]) {
	if (argc > 2 && strcmp(argv[1], "run_tests") == 0) {
		//run_tests(std::cout);
		return 0;
	}

	feature_set fs;
	load_feature_set(fs, "training_set.txt");
	fs.print(std::cout);

    size_t n = fs.feature_count();
    size_t M = 5; // number of trees
    //size_t M = 7; the last two splits for 0 - all samples

    // learning
    ensemble<decision_stump, gradient_boosting> e(M);
    e.learn(fs);
    e.print(std::cout);

    // prediction
    std::unique_ptr<double[]> x(new double[n]);
    e.predict(x.get(), n);
	return 0;
}

void run_tests(std::ostream& os) {
	// todo: 
}
