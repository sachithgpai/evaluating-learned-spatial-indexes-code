

#include<vector>
#include<iostream>
#include<fstream>
#include<chrono>
#include<string>
#include<algorithm>
#include<cmath>
#include <filesystem>

#include"../Indexes/utils/density_estimators/dens_est.h"


using namespace std;



vector<size_t> BlockSizes{32, 64, 128, 256, 512, 1024, 2048, 4096};
vector<string> selectivities_arr{"00064", "00256", "01024", "04096", "16384"};


void write_countbased_query(ofstream& fout, size_t& num_required, Point& query_center, DensEstTree& dens_est_obj){
    
    // cout<<"write_countbased_query  num_required:"<<num_required<<" point: "; query_center.Print();
    double_t low=0.00001,high=1.0,mid,query_count_estimate ;
    BoundingRectangle mbr;

    while((high-low)>0.00001){
        mid = (low+high)/2.0;

        mbr.low_.elements_[0] = max(0.0,(query_center.elements_[0]-mid));
        mbr.low_.elements_[1] = max(0.0,(query_center.elements_[1]-mid));

        mbr.high_.elements_[0] = min(1.0,(query_center.elements_[0]+mid));
        mbr.high_.elements_[1] = min(1.0,(query_center.elements_[1]+mid));

        query_count_estimate = dens_est_obj.EstimateCount(mbr);

        if(abs(num_required-query_count_estimate)<0.01*num_required)
            break;
        if(query_count_estimate>num_required)       high = mid;
        else if(query_count_estimate<num_required)       low = mid;
    }
    fout<<mbr.low_.elements_[0]<<" "<<mbr.low_.elements_[1]<<" "<<mbr.high_.elements_[0]<<" "<<mbr.high_.elements_[1]<<std::endl;
}


int main(int argc, char* argv[]){

    string dataset_folder_name= string(argv[1]);
    int num_samples_per_group = 5;


    for(int data_sample_num = 1;data_sample_num<=num_samples_per_group;data_sample_num++){
        for(int data_ent_id = 1;data_ent_id<=num_samples_per_group;data_ent_id++){
            cout<<"data_sample_num: "<<data_sample_num<<" data_ent_id: "<<data_ent_id<<std::endl;
            vector<Point> datapoints;
            double_t a, b, c, d;
            ifstream pointsfile(PROJECT_ROOT+"Datasets/"+dataset_folder_name+"/"+to_string(data_sample_num)+"/datapoints/"+to_string(data_ent_id));
            while (pointsfile >> a >> b)
                datapoints.push_back(Point(a,b));
            pointsfile.close();

            size_t N = datapoints.size();

            DensEstTree dens_est_obj(datapoints, 128, BoundingRectangle(Point(0.0,0.0),Point(1.0,1.0)));

            for(int query_ent_id = 1;query_ent_id<=num_samples_per_group;query_ent_id++){
                vector<Point> queries_centers;
                ifstream queriesfile(PROJECT_ROOT+"Datasets/"+dataset_folder_name+"/"+to_string(data_sample_num)+"/queries/otherDist/"+to_string(data_ent_id)+"_"+selectivities_arr[0]+"_querycenteres_"+to_string(query_ent_id));
                while (queriesfile >> a >> b)
                    queries_centers.push_back(Point(a,b));
                queriesfile.close();


                for(string selectivity: selectivities_arr){
                    size_t num_required = size_t((N*std::stof(selectivity)/1000000));
                    ofstream count_queriesfile(PROJECT_ROOT+"Datasets/"+dataset_folder_name+"/"+to_string(data_sample_num)+"/queries/otherDist/"+to_string(data_ent_id)+"_"+selectivity+"_countbased_"+to_string(query_ent_id),ios_base::app);

                    for(Point& query_center: queries_centers)
                        write_countbased_query(count_queriesfile,num_required,query_center,dens_est_obj);
                    
                    count_queriesfile.close();
                    cout<<PROJECT_ROOT+"Datasets/"+dataset_folder_name+"/"+to_string(data_sample_num)+"/queries/otherDist/"+to_string(data_ent_id)+"_"+selectivity+"_countbased_"+to_string(query_ent_id)<<std::endl;  

                }
            }
        }
    }
    return 0;
}