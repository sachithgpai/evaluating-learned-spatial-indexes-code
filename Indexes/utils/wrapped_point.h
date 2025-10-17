#ifndef WRAPPEDPOINT_H
#define WRAPPEDPOINT_H
#include <vector>
#include <string>
#include"point.h"


/* A class to wrap floating point dataset so as to project them into rank space and potentially use space filling curves.
*/
class WrappedPoint : public Point
{
public:

    uint32_t rank_elements_[Constants::DIM];
    uint64_t curve_value_;
    double_t num_queries_overlapping_;          // default should be 0.5 to make sure the points that are never included in queries are also weighted.
    double_t temp_cum_num_queries_overlapping_{};

    WrappedPoint(){ rank_elements_[0]=0; rank_elements_[1]=0; curve_value_=0; num_queries_overlapping_=0.5; }
    WrappedPoint(const Point &other):Point(other){ rank_elements_[0]=0; rank_elements_[1]=0; curve_value_=0; num_queries_overlapping_=0.5; }
    Point ExtractPoint(){ return Point(elements_[0],elements_[1]); }

    void Print(){
        std::cout<<"(";
        for(auto i=0;i<Constants::DIM;i++)
            std::cout<< std::setprecision(9)<<elements_[i]<<", ";
        std::cout<<")  (";
        for(auto i=0;i<Constants::DIM;i++)
            std::cout<<rank_elements_[i]<<", ";
        std::cout<<")  "<<curve_value_<<"\n";
    }
};

#endif
