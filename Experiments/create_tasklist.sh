foldername=$1


rm hq_tasks_* 
for data_ent_id in {1..5}  
do
    for data_sample_num in {1..5} 
    do
        for BLOCK_SIZE in 32 64 128 256 512 1024 2048 4096 
        do
            for selectivity_id in {0..4} #indicates the selectivity of query worload
            do
                for  query_ent_id in {1..5} # query entropy id
                do
                    echo "/scratch/project_2005865/sachithp/experiments-md-index/Experiments/build_evaluate.out ${foldername} ${data_sample_num} ${data_ent_id} ${BLOCK_SIZE} ${query_ent_id} ${selectivity_id}"  >> hq_tasks_evaluate

                done
            done
            echo "python /scratch/project_2005865/sachithp/experiments-md-index/Indexes/RTree/RSMI.py ${foldername} ${data_sample_num} ${data_ent_id} ${BLOCK_SIZE}"  >> hq_tasks_RSMI
        done
    done
done


# Creating folders for intermediate index storage and results.
mkdir -p $foldername/TrainedIndexes $foldername/ResultsFolder
cd $foldername/TrainedIndexes
mkdir QDTree RSMI FLOOD
cd ../..

# Creating a temporary disk back folder
mkdir temp_blockstore