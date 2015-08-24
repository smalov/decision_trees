#include "feature_set.h"

// the last column is label
void load_feature_set(feature_set& fs, const char* file_name, bool classification) {
	std::ifstream f(file_name);
	std::string line;
	// read header
	if (!std::getline(f, line))
		throw std::exception("empty file");
	size_t n = 0; // number of columns
	size_t first = 0, last = 0;
	for (; last != std::string::npos;) {
		last = line.find('\t', first);
		first = last + 1;
		++n;
	}
	// read values
	feature_data data;
	try {
		while (std::getline(f, line)) {
			data.push_back(new double[n]);
			first = 0, last = 0;
			size_t i = 0;
			while (i < n) {
				last = line.find('\t', first);
				double val = strtod(line.c_str() + first, nullptr);
				if (i == (n - 1) && classification) {
					if (val == 0.0)
						val = -1.0;
					else if (val != 1.0)
						throw std::exception("unexpected label value");
				}
				data.back()[i++] = val;
				if (last == std::string::npos)
					break;
				first = last + 1;
			}
			if (i != n)
				throw std::exception("invalid number of values");
		}
	} catch (const std::exception& e) {
		delete_data(data);
	}
	fs.initialize(data, n);
}

void load_feature_set(feature_set& fs, const char* file_name) {
	load_feature_set(fs, file_name, false);
}

void load_feature_set_for_classification(feature_set& fs, const char* file_name) {
	load_feature_set(fs, file_name, true);
}

void print_data(std::ostream& os, const feature_data& data, size_t n) {
	std::cout << "feature vectors:\n";
	for (size_t i = 0; i < data.size(); ++i) {
		const double* x = data[i];
		os << (i + 1) << ":";
		for (size_t j = 0; j <= n; ++j)
			os << "\t" << std::setprecision(3) << x[j];
		os << "\n";
	}
	os << std::endl;
}

void delete_data(feature_data& data) {
	for (size_t i = 0; i < data.size(); ++i)
		delete[] data[i];
}
