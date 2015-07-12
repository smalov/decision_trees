#include <iostream>
#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include "regression_tree.h"
#include "feature_set.h"
#include "ensemble.h"

int main(int argc, char* argv[]) {
	feature_set_ptr fs(load_features("training_set.txt"));
	fs->print(std::cout);

    size_t n = fs->get_feature_count();
    size_t M = 5; // number of trees
    //size_t M = 7; the last two splits for 0 - all samples

    // learning
    ensemble e(M);
    e.learn(*fs);
    e.print(std::cout);

    // prediction
    std::vector<double> x(n);
    e.predict(x, n);
	return 0;
}

