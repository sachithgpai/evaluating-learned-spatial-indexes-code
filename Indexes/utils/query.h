#ifndef QUERY_H
#define QUERY_H

#include<cmath>
#include<functional>
#include<limits.h>
#include<math.h>
#include"bounding_rectangle.h"


class Query: public BoundingRectangle{
    public:
        bool dim_used_[Constants::DIM];

        Query(){}

        /** Construct a query from its low/high corner points. */
        Query(const Point& a, const Point& b):BoundingRectangle(a,b){}

        /** Convenience constructor for 2D axis-aligned query bounds. */
        Query(const double_t& a, const double_t& b,const double_t& c, const double_t& d):BoundingRectangle(Point(a,b),Point(c,d)){}


        /** Count how many dimensions are active in the query mask. */
        int NumDim(){
            int result=0; 
            for(size_t i=0;i<Constants::DIM;i++)
                if(dim_used_[i])
                    result++;
            return result;
        }



};

#endif
