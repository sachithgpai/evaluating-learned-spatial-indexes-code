#ifndef CONSTANTS_H
#define CONSTANTS_H
using namespace std;

#include<cstdlib>
#include<string>

/**
 * Global experiment-time block size. The evaluator mutates this value before
 * constructing indexes so all implementations observe the same page budget.
 */
size_t BLOCK_SIZE = 256;



/** Branching-factor defaults for the R-tree variants. */
size_t STR_BRANCH_FACTOR = 16;
size_t RSTAR_BRANCH_FACTOR = 16;
size_t CUR_BRANCH_FACTOR = 16;
size_t RW_BRANCH_FACTOR = 16;

std::string NormalizeProjectRoot(std::string path)
{
    if(!path.empty() && path.back() != '/')
        path += "/";
    return path;
}

/** Absolute path to the repository root used by the experiment entrypoints. */
std::string PROJECT_ROOT = []() {
    const char* project_root = std::getenv("PROJECT_ROOT");
    if(project_root != nullptr && std::string(project_root).size() > 0)
        return NormalizeProjectRoot(std::string(project_root));
    return NormalizeProjectRoot(
        "/scratch/project_2005865/sachithp/EvalPublicRepo/evaluating-learned-spatial-indexes/"
    );
}();

/**
 * Shared compile-time constants used across the index implementations.
 */
class Constants
{
public:

    static constexpr double_t EPSILON_ERR = 1e-9;

    static constexpr size_t DIM = 2;
    static constexpr size_t LEAF_SORT_DIM = 0;      // this values should be in range [0,DIM) 

    Constants();
};

#endif
