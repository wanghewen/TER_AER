This code is used for running experiments for the paper. 
1. Download the dataset file from [https://mega.nz/file/wuoVTRJT#LejeGzAkdRTQu3AKOM-hYuAHwM0eg1pVDBT2AQ4nxF8](https://mega.nz/file/wuoVTRJT#LejeGzAkdRTQu3AKOM-hYuAHwM0eg1pVDBT2AQ4nxF8) and unzip
2. Install all dependencies in the _requirements.txt_
3. Run the following command to get all results: `export PYTHONPATH=.;python train_EdgeAttributed.py --data_folder 
   "$(pwd)/data" | tee run.log`
4. Check the results in _run.log_

Note that it will take some time to generate the cache files when you run for the first time.

Reference: 

Hewen Wang, Renchi Yang, Keke Huang, and Xiaokui Xiao. 2023. Efficient and Effective Edge-wise Graph Representation Learning. In Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '23). Association for Computing Machinery, New York, NY, USA, 2326–2336. https://doi.org/10.1145/3580305.3599321
