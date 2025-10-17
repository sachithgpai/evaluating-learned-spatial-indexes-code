#ifndef CONSTANTS_H
#define CONSTANTS_H
using namespace std;

#include<string>

size_t BLOCK_SIZE = 256;



size_t STR_BRANCH_FACTOR = 16;
size_t RSTAR_BRANCH_FACTOR = 16;
size_t CUR_BRANCH_FACTOR = 16;
size_t RW_BRANCH_FACTOR = 16;

std::string PROJECT_ROOT = "/scratch/project_2005865/sachithp/experiments-md-index/";

class Constants
{
public:

    static constexpr double_t EPSILON_ERR = 1e-9;

    static constexpr size_t DIM = 2;
    static constexpr size_t LEAF_SORT_DIM = 0;      // this values should be in range [0,DIM) 

    Constants();
};

#endif