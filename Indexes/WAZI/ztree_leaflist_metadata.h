#ifndef ZTREE_LOCALMODEL_H
#define ZTREE_LOCALMODEL_H


#include<algorithm>
#include<vector>
#include<list>
#include"../utils/local_model.h"


/**
 * Metadata stored for each leaf in the ZTree leaf list.
 */
class ZtreeLeaflistMetadata{
    public:
        BoundingRectangle mbr_;
        uint64_t page_mask_;                            // Instead of having pages, we can assign each page a 'mask' based on Z-order like bit flip at each level.
        size_t fwd_iter_1_, fwd_iter_2_, fwd_iter_3_, fwd_iter_4_;



        /** Default-construct an empty metadata record. */
        ZtreeLeaflistMetadata(){}

        /** Seed the metadata with the leaf MBR. */
        ZtreeLeaflistMetadata(BoundingRectangle mbr):mbr_(mbr){}

        /** Initialize all forward iterators to the current leaf position. */
        void InitializeFwdIters(size_t id){
            fwd_iter_1_ = id;             
            fwd_iter_2_ = id;
            fwd_iter_3_ = id;
            fwd_iter_4_ = id;
        }

        /** Encode which directional skip cases apply for `query`. */
        uint8_t CheckFwdIterCases(const Query query){
            uint8_t case_mask =0;
            if(mbr_.high_.elements_[0]<query.low_.elements_[0]) case_mask |= 1;
            if(mbr_.high_.elements_[1]<query.low_.elements_[1]) case_mask |= 2;
            if(mbr_.low_.elements_[0]>query.high_.elements_[0]) case_mask |= 4;
            if(mbr_.low_.elements_[1]>query.high_.elements_[1]) case_mask |= 8;
            return case_mask;
        }

        /** Print the leaf-list metadata for debugging. */
        void Print(){
            std::cout<<" LocalModel (";
            for(auto i=0;i<Constants::DIM;i++)
                std::cout<<mbr_.low_.elements_[i]<<((i+1==Constants::DIM)?" ) ":" , ");

            std::cout<<" (";
            for(auto i=0;i<Constants::DIM;i++)
                std::cout<<mbr_.high_.elements_[i]<<((i+1==Constants::DIM)?" ) ":" , ");

            std::cout<<" page_mask:"<<page_mask_;

            std::cout<<"\n";
        }
};




#endif
