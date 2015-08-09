#pragma once

template <typename Tree>
class no_pruning {
public:
	no_pruning() {}
	void prune(Tree& t) {}
};
