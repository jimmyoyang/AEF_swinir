export PATH=/mnt/lm_data_afs/wangzining/charles/miniconda3/envs/alphaearth/bin:$PATH
export PYTHON=/mnt/lm_data_afs/wangzining/charles/miniconda3/envs/alphaearth/bin/python
export PIP=/mnt/lm_data_afs/wangzining/charles/miniconda3/envs/alphaearth/bin/pip

rm data raw_landsat training_logs files
ln -s /mnt/lm_data_afs/wangzining/charles/datasets/1652288917_charles2530/raw_landsat data
ln -s /mnt/lm_data_afs/wangzining/charles/datasets/1652288917_charles2530/raw_landsat/raw_landsat raw_landsat
ln -s /mnt/lm_data_afs/wangzining/charles/logs/AEF_swinir/ training_logs
ln -s /mnt/lm_data_afs/wangzining/charles/files/AEF_files files