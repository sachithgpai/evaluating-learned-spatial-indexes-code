/**
 * @file point.h
 * @author Sachith (sachith.pai@helsinki.fi)
 * @brief A template point class taking variadic number of arguments.
 * @version 0.1
 * @date 2023-07-25
 * 
 * @copyright Copyright (c) 2023
 * 
 */


#ifndef POINT_H
#define POINT_H

#include<iostream>
#include<cmath>
#include<algorithm>
#include"constants.h"

/**
 * @brief The basic point in our setting.
 *         NOTE: if you have a template for n D and pass it k<n arguments there wont be any errors.
 */

class Point{
    public:
    double_t elements_[Constants::DIM];

    template <typename ... Args>
    Point(const Args& ... args) : elements_{args...} {}

    Point(){}

    Point(const Point &other){
        std::copy(std::begin(other.elements_), std::end(other.elements_), std::begin(elements_));
    }

    void Print(){
        for(auto i=0;i<Constants::DIM;i++)
            std::cout<<elements_[i]<<" ";
        std::cout<<"\n";
    }

    bool operator==(const Point& other_pnt) const{
        bool result = true;
        for(size_t i=0;i<Constants::DIM;i++)
            result &= (elements_[i]==other_pnt.elements_[i]);
        return result;
    }

};

/**
 * @brief Comparator function for the sorting Point data in a vector.
*/
template<int SortDimension>
bool SortOrder(Point i,Point j) { return i.elements_[SortDimension]<j.elements_[SortDimension]; }



bool FullSortOrder(Point a,Point b,int *dim_order) { 
    for(int i=0;i<Constants::DIM;i++)
        if (a.elements_[dim_order[i]]<b.elements_[dim_order[i]]) 
            return true; 
    return false;
}



class PaddedPoint: public Point
{
    public:
    char pad[128];

    PaddedPoint(const Point &other):Point(other){}

};



#endif
