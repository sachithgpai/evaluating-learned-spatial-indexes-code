/**
 * @file bounding_rectangle.h   
 * @author Sachith (sachith.pai@helsinki.fi)
 * @brief A bounding rectangle instead of a point.
 * @version 0.1
 * @date 2022-04-25
 * 
 * @copyright Copyright (c) 2022
 * 
 */


#ifndef RECT_H
#define RECT_H

#include<cmath>
#include<functional>
#include<limits>
// #include<math.>
#include"point.h"


class BoundingRectangle{
    public:
    Point low_;
    Point high_;

    /** Create an empty rectangle whose bounds can be expanded incrementally. */
    BoundingRectangle(){ 
        SetToDefault();
    }
    
    /** Construct a rectangle from low/high corner points. */
    BoundingRectangle(const Point &x,const Point &y): low_(x), high_(y){ }

    /** Return true when the two rectangles have a non-empty overlap. */
    bool IsThereOverlap(const BoundingRectangle& other_mbr){
        bool result = true;
        for(size_t i =0;i<Constants::DIM;i++)
            result &= (std::max(low_.elements_[i],other_mbr.low_.elements_[i])<std::min(high_.elements_[i],other_mbr.high_.elements_[i]));
        return result;
    }


    /** Return true when this rectangle fully covers `other_mbr`. */
    bool IsCompletelyCovering(const BoundingRectangle& other_mbr){
        bool result = true;
        for(size_t i =0;i<Constants::DIM;i++)
            result &= (low_.elements_[i]<=other_mbr.low_.elements_[i]) && (high_.elements_[i]>other_mbr.high_.elements_[i]);
        return result;
    }    

    /** Check whether a point lies inside the rectangle. */
    bool CheckPointWithin(const Point& point){

        bool result = true;
        for(size_t i =0;i<Constants::DIM;i++)
            result &= (point.elements_[i] >= low_.elements_[i] && point.elements_[i]<high_.elements_[i]);
        
        return result;
    }


    /** Expand the bounds to span the full floating-point domain. */
    void SetToSpanWholeSpace(){
        std::fill_n(low_.elements_,Constants::DIM, std::numeric_limits<double_t>::min());
        std::fill_n(high_.elements_,Constants::DIM, std::numeric_limits<double_t>::max());        
    }

    /** Reset the rectangle to an empty state before incremental updates. */
    void SetToDefault(){
        std::fill_n(low_.elements_,Constants::DIM, std::numeric_limits<double_t>::max());
        std::fill_n(high_.elements_,Constants::DIM, std::numeric_limits<double_t>::min());        
    }

    /** Compare the low/high bounds exactly. */
    bool operator==(const BoundingRectangle& other_mbr){  

        bool result = true;
        for(size_t i =0;i<Constants::DIM;i++)
            result &= (low_.elements_[i] == other_mbr.low_.elements_[i] && high_.elements_[i] ==other_mbr.high_.elements_[i]);
        
        return result;
    }

    /** Return the hyper-rectangle area/volume. */
    double_t Area(){
        double_t result = 1;
        for(size_t i =0;i<Constants::DIM;i++)
            result *= (high_.elements_[i]-low_.elements_[i]);
        return result;
    }

    /** Return overlap area divided by this rectangle's area. */
    double_t RatioOfOverlap(const BoundingRectangle& other_mbr){
        // Area of overlap between two BRs over area of curent BR
        return AreaOfOverlap(other_mbr)/Area();
    }

    /** Compute the overlap area between two rectangles. */
    double_t AreaOfOverlap(const BoundingRectangle& other_mbr){
        double_t area_of_overlap = 1.0;
        for(int i=0;i<Constants::DIM;i++){
            double_t overlap =
                std::min(high_.elements_[i],other_mbr.high_.elements_[i]) -
                std::max(low_.elements_[i],other_mbr.low_.elements_[i]);
            if(overlap<=0.0)
                return 0.0;
            area_of_overlap *= overlap;
        }
        return area_of_overlap;
    }

    /** Expand the rectangle to include a new point. */
    void UpdateBoundingBoxWithPoint(const Point &pnt){
        for(size_t i=0;i<Constants::DIM;i++){
            low_.elements_[i] = std::min(low_.elements_[i],pnt.elements_[i]);
            high_.elements_[i] = std::max(high_.elements_[i],pnt.elements_[i]);//+Constants::EPSILON_ERR);
        }
    }

    /** Expand the rectangle to include another rectangle. */
    void UpdateBoundingBoxWithBoundingBox(const BoundingRectangle &other_mbr){
        for(size_t i=0;i<Constants::DIM;i++){
            low_.elements_[i] = std::min(low_.elements_[i],other_mbr.low_.elements_[i]);
            high_.elements_[i] = std::max(high_.elements_[i],other_mbr.high_.elements_[i]);
        }
    }

    /** Print the rectangle bounds for debugging. */
    void Print(){
        std::cout<<"(";
        for(auto i=0;i<Constants::DIM;i++)
            std::cout<<low_.elements_[i]<<", ";

        std::cout<<") x (";
        for(auto i=0;i<Constants::DIM;i++)
            std::cout<<high_.elements_[i]<<", ";
        std::cout<<") "<<"\n";
    }

    /** Return the perimeter of the rectangle. */
    double_t Perimeter(){
        double_t result = 0;
        for(size_t i =0;i<Constants::DIM;i++)
            result += 2*(high_.elements_[i]-low_.elements_[i]);
        return result;
    }

    /** Estimate how much the perimeter would grow after inserting a point. */
    double_t DeltaPerimeterOnUpdate(const Point& pnt){
        double_t result = 0;
        for(size_t i =0;i<Constants::DIM;i++)
            result += ( std::abs(high_.elements_[i]-pnt.elements_[i]) 
                        + std::abs(low_.elements_[i]-pnt.elements_[i])
                        - (high_.elements_[i]-low_.elements_[i]) );
        return result;
    }

    /** Estimate how much the area would grow after inserting a point. */
    double_t DeltaAreaOnUpdate(const Point& pnt){
        double_t result = 1;
        for(size_t i =0;i<Constants::DIM;i++)
            result *=  0.5* ( std::abs(high_.elements_[i]-pnt.elements_[i]) 
                        + std::abs(low_.elements_[i]-pnt.elements_[i])
                        + (high_.elements_[i]-low_.elements_[i]) );
        return result;
    }

};

#endif
