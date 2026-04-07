#ifndef SORT_TOOLS_H
#define SORT_TOOLS_H

#include <algorithm>
#include <array>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <cstdlib>
#include"point.h"
#include"query.h"
#include"wrapped_point.h"
#include"./pgm/pgm_index.hpp"

#define SortX 0
#define SortY 1

/** Lower-bound style binary search over a sorted vector. */
template<typename T>
int BinarySearch(std::vector<T> arr, T val){
  int first = 0, last=arr.size()-1,mid;
  while(first<last){
    mid = (first+last)/2;

    if(arr[mid]>val) last=mid;
    if(arr[mid]<val) first = mid+1;
  }
  return last;

}
/** Lower-bound style binary search restricted to `[first, last]`. */
template<typename T>
int32_t BinarySearch(std::vector<T> arr, T val,int32_t first, int32_t last){
  // std::cout<<"BinarySearch val "<<val<<" first "<<first<<" last "<<last<<"\n";
  int mid;
  while(first<last){
    mid = (first+last)/2;
    // std::cout<<"BinarySearch mid "<<mid<<"["<<arr[mid]<<"] first "<<first<<"["<<arr[first]<<"] last"<<last<<"["<<arr[last]<<"]"<<"\n";
    // std::cin.get();
    if(arr[mid]>=val) last=mid;
    if(arr[mid]<val) first = mid+1;
    
  }
  return last;

}


/** Comparator that sorts `Point` instances by one dimension. */
class SortOrderer {
  size_t i;
public:
  SortOrderer(size_t i) : i{i}{}
  [[nodiscard]] constexpr bool operator()(const Point& a, const Point& b) const noexcept {
    return a.elements_[i] < b.elements_[i];
  }
};


/** Comparator that sorts wrapped points by one geometric dimension. */
class WrappedPointSortOrderer {
  size_t i;
public:
  WrappedPointSortOrderer(size_t i) : i{i}{}
  [[nodiscard]] constexpr bool operator()(const WrappedPoint& a, const WrappedPoint& b) const noexcept {
    return a.elements_[i] < b.elements_[i];
  }
};

/** Comparator that sorts wrapped points by one rank-space dimension. */
class WrappedPointRankOrderer {
  size_t i;
public:
  WrappedPointRankOrderer(size_t i) : i{i}{}
  [[nodiscard]] constexpr bool operator()(const WrappedPoint& a, const WrappedPoint& b) const noexcept {
    return a.rank_elements_[i] < b.rank_elements_[i];
  }
};

/** Comparator that sorts wrapped points by Morton/Z-order value. */
class WrappedPointCurveValueOrder {
    public:
    bool operator()(const WrappedPoint& a, const WrappedPoint& b)
    {
        return a.curve_value_ < b.curve_value_;
    }
};
/**
 * Predicate used with `std::partition` to split points around one dimension.
 */
class ComparatorPointPartition {
    public:
    Point orig_;
    size_t d_;
    
    ComparatorPointPartition(){}
    ComparatorPointPartition(const Point &point, const size_t d):orig_(point),d_(d){ }
    ComparatorPointPartition(const double_t &split_value, const size_t d):orig_(split_value,split_value),d_(d){ }
    bool operator()(const Point& a)
    {
        return (a.elements_[d_]<orig_.elements_[d_]);
    }

    bool operator<(const Point& a)
    {
        return (a.elements_[d_]<orig_.elements_[d_]);
    }

    void SetDimSplitValue(size_t d,double_t split){ orig_.elements_[d]=split; d_=d;}
};
/**
 * Approximate rank-space mapper backed by PGM indexes on each dimension.
 */
class RankSpaceMapper{
  public:
    /* Settting the default epsilon paramter (1024)*/
    pgm::PGMIndex<double_t, 128>* pgm_index_0;
    pgm::PGMIndex<double_t, 128>* pgm_index_1;
    std::vector<double_t> sorted_0dim;
    std::vector<double_t> sorted_1dim;


    /** Build one PGM index per dimension over the sorted coordinates. */
    RankSpaceMapper(std::vector<Point>& data){
      sorted_0dim.reserve(data.size());
      for(auto& pnt: data) sorted_0dim.push_back(pnt.elements_[0]);
      std::sort(sorted_0dim.begin(),sorted_0dim.end());
      pgm_index_0 = new pgm::PGMIndex<double_t, 128>(sorted_0dim);

      sorted_1dim.reserve(data.size());
      for(auto& pnt: data) sorted_1dim.push_back(pnt.elements_[1]);
      std::sort(sorted_1dim.begin(),sorted_1dim.end());
      pgm_index_1 = new pgm::PGMIndex<double_t, 128>(sorted_1dim);

    }

    /** Transform a geometric point into its wrapped rank-space form. */
    WrappedPoint Transform(const Point& a){
        // std::cout<<"Transform"<<"\n";
        WrappedPoint wrapped_point(a);
        auto range0 = pgm_index_0->search(a.elements_[0]);
        // std::cout<<"Transform  pgm_index_0->search"<<"\n";

        wrapped_point.rank_elements_[0] = BinarySearch<double_t>(sorted_0dim, a.elements_[0], range0.lo, range0.hi-1);
        // std::cout<<"Transform BinarySearch"<<"\n";

        auto range1 = pgm_index_1->search(a.elements_[1]);
        // std::cout<<"Transform  pgm_index_1->search"<<"\n";

        wrapped_point.rank_elements_[1] = BinarySearch<double_t>(sorted_1dim, a.elements_[1], range1.lo, range1.hi-1);
        // std::cout<<"Transform BinarySearch"<<"\n";

        // wrapped_point.Print();
        // std::cin.get();
        return wrapped_point;
    }
    /** Update an existing wrapped point in place with rank coordinates. */
    void Transform(WrappedPoint& wrp_pt){
        // std::cout<<"Transform"<<"\n";
        auto range0 = pgm_index_0->search(wrp_pt.elements_[0]);
        // std::cout<<"Transform  pgm_index_0->search"<<"\n";

        wrp_pt.rank_elements_[0] = BinarySearch<double_t>(sorted_0dim, wrp_pt.elements_[0], range0.lo, range0.hi-1);
        // std::cout<<"Transform BinarySearch"<<"\n";

        auto range1 = pgm_index_1->search(wrp_pt.elements_[1]);
        // std::cout<<"Transform  pgm_index_1->search"<<"\n";

        wrp_pt.rank_elements_[1] = BinarySearch<double_t>(sorted_1dim, wrp_pt.elements_[1], range1.lo, range1.hi-1);
        // std::cout<<"Transform BinarySearch"<<"\n";

    }

    /** Rewrite wrapped points so each dimension has dense integer ranks. */
    void Reintegerize(std::vector<WrappedPoint>::iterator vec_begin,std::vector<WrappedPoint>::iterator vec_end){
      std::sort(vec_begin,vec_end,WrappedPointSortOrderer(0));
      uint32_t ix=0;
      for(auto iter = vec_begin; iter!=vec_end; iter++) iter->rank_elements_[0]=ix++;
      
      
      std::sort(vec_begin,vec_end,WrappedPointSortOrderer(1));
      ix=0;
      for(auto iter = vec_begin; iter!=vec_end; iter++) iter->rank_elements_[1]=ix++;

      // for(auto iter = vec_begin; iter!=vec_end; iter++) Transform(*iter);
      
    }

};


/** Comparator that orders queries by one lower-bound coordinate. */
class QuerySortOrderer {
  size_t i;
public:
  QuerySortOrderer(size_t i) : i{i}{}
  [[nodiscard]] constexpr bool operator()(const Query& a, const Query& b) const noexcept {
    return a.low_.elements_[i] < b.low_.elements_[i];
  }
};

/** Comparator that orders queries by one upper-bound coordinate. */
class QuerySortOrdererUpperBound {
  size_t i;
public:
  QuerySortOrdererUpperBound(size_t i) : i{i}{}
  [[nodiscard]] constexpr bool operator()(const Query& a, const Query& b) const noexcept {
    return a.high_.elements_[i] < b.high_.elements_[i];
  }
};

/**
 * Wrap points and annotate each with the number of queries that contain it.
 *
 * TODO: Improve efficiency; this currently scans the candidate point range for
 * each query similarly to the CUR-tree weighting pass.
 */
std::vector<WrappedPoint> WeightPointsWithQuery(std::vector<Point> data, std::vector<Query>& queries){

    WrappedPointSortOrderer x_sorter(0);

    std::vector<WrappedPoint> wrapped_data;
    for(auto& pnt: data) wrapped_data.push_back(WrappedPoint(pnt)); 
    std:sort(wrapped_data.begin(),wrapped_data.end(),x_sorter);

    for(auto& q: queries){
        auto data_iter_low = std::lower_bound(wrapped_data.begin(), wrapped_data.end(), q.low_ , x_sorter);
        auto data_iter_high = std::upper_bound(wrapped_data.begin(), wrapped_data.end(), q.high_ , x_sorter);

        for(auto it= data_iter_low;it!=data_iter_high;it++)
            if(q.CheckPointWithin(it->ExtractPoint()))
                it->num_queries_overlapping_+=1;
        
    }
    return wrapped_data;
}

/** Compute the Morton/Z-order value from the wrapped rank coordinates. */
uint64_t compute_Z_value(WrappedPoint& wrap_pt)
{
	uint64_t result = 0;
	for (int i = 0; i < 31; i++)
	{
		uint64_t seed = (uint64_t) pow(2, i);

		uint64_t temp = seed & wrap_pt.rank_elements_[0];
		temp = temp << i;
		result += temp;

		temp = seed & wrap_pt.rank_elements_[1];
		temp = temp << (i+1);
		result += temp;
	}
	return result;
}

/** Construct a well-seeded random engine of type `T`. */
template <typename T = std::mt19937>
auto random_generator() -> T {
    auto constexpr seed_bytes = sizeof(typename T::result_type) * T::state_size;
    auto constexpr seed_len = seed_bytes / sizeof(std::seed_seq::result_type);
    auto seed = std::array<std::seed_seq::result_type, seed_len>();
    auto dev = std::random_device();
    std::generate_n(begin(seed), seed_len, std::ref(dev));
    auto seed_seq = std::seed_seq(begin(seed), end(seed));
    return T{seed_seq};
}

/** Generate a random alphanumeric string of the requested length. */
auto generate_random_alphanumeric_string(std::size_t len) -> std::string {
    static constexpr auto chars =
        "0123456789"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz";
    thread_local auto rng = random_generator<>();
    auto dist = std::uniform_int_distribution{{}, std::strlen(chars) - 1};
    auto result = std::string(len, '\0');
    std::generate_n(begin(result), len, [&]() { return chars[dist(rng)]; });
    return result;
}

#endif
