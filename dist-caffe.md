## Distributed Caffe Notes

#### Old Caffe
Entrance: `train()` at `caffe_main.cpp`
- `caffe_engine(new caffe::CaffeEngine<float>(solver_param))`
    - `CaffeEngine::Init(param)`
        - `CaffeEngine::InitPS()`
            - `InitPSForTrainNet(num_test_nets + 1)`
                - `net_->InitPS(net_param, true, num_additional_tables, &layer_blobs_global_idx_)`
                    - `PSTableGroup::Init(table_group_config, false);`
                    - `for layers: layers_[layer_id]->SetUp`
                        - `concrete class: LayerSetUp()`
                            - `for blobs: blob[id]->CreatePSTable()`
                                - **CreateTable** `PSTableGroup::CreateTable(global_id_, table_config);` 
            - `InitPSForTestNets(num_test_nets)`
                - `net_->InitPS(net_param, true, num_additional_tables, &layer_blobs_global_idx_)`
                - `...`
- `petuum::PSTableGroup::CreateTableDone();`
- `thread(&caffe::CaffeEngine<float>::Start, std::ref(*caffe_engine))`
    - `new Solve()`
        - `InitTrainNet()`
            - **upload params** `set_table: outputs_global_table_`
            - **upload params** `layer->SetUpBlobGlobalTable`.
                - `blobs_[idx]->set_table`
                - `FillPSTable(this->blobs_[0].get())`
    - `Solve():`
        - `for (; iter_ < param_.max_iter(); ++iter_)`
            - `JoinSyncThreads();`
            - `ForwardBackward(bottom_vec)`
                - `for (int i = layers.size() - 1; i >= 0; --i)`
                    - `new std::thread(&Solver::ThreadSyncWithPS, this, ...)`
                        - `Blob::UpdatePSTable(param->cpu_diff())`
                        - `Blob::SyncWithPSTable(clock + 1)`
