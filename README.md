# van_depth
visual attention network based monocular depth estimation

#### install
```
conda create -n van_depth python=3.7 -y 
conda activate van_depth
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -y #pytorch1.11.0
pip install opencv-python==4.5.2.52 tensorboardX==1.5 scikit-image==0.16.2 timm==0.4.12
pip install tqdm==4.57.0 xlrd==2.0.1 xlwt==1.3.0
```

#### prepare kitti dataset
Following [monodepth2](https://github.com/nianticlabs/monodepth2#-kitti-training-data)

#### download pretrained [VAN](https://github.com/Visual-Attention-Network/VAN-Classification)
```
mkdir pretrained
wget -P ./pretrained/ https://huggingface.co/Visual-Attention-Network/VAN-Small-original/resolve/main/van_small_811.pth.tar
```

#### train on kitti
```
CUDA_VISIBLE_DEVICES=0 python kitti_train.py --data_path ${kitti_data_path} --log_dir ./kitti_logs --model_name "kitti_ms_models"  --frame_ids 0 -1 1 --scales 0 2 3 4 --use_stereo
```

#### test on the eigen split of kitti
```
CUDA_VISIBLE_DEVICES=0 python evaluate_kitti_depth.py --scales 0 2 3 4 --data_path ${kitti_data_path} --load_weights_folder "${model_weight}" --eval_stereo
```
The test results of our model (baidunetdisk https://pan.baidu.com/s/14ohVPXOnWKj7krq4N7ycsA, code: 5wh7) trained on kitti are as follows:
   abs_rel |   sq_rel |     rmse | rmse_log |       a1 |       a2 |       a3 | 
&   0.099  &   0.775  &   4.570  &   0.186  &   0.893  &   0.961  &   0.981  \\

#### prepare seasondepth dataset
donload and unzip [train set](https://doi.org/10.6084/m9.figshare.16442025) ans [val set](https://doi.org/10.6084/m9.figshare.14731323)
Prepare [test set](http://seasondepth-challenge.org/index/static/dataset/ICRA2022_SeasonDepth_Test_RGB.zip) following [seasondepth](https://github.com/SeasonDepth/SeasonDepth/tree/master/dataset_info)

#### train on seasondepth
```
CUDA_VISIBLE_DEVICES=6 python season_train.py --set_seed --split seasondepth --height 384 --width 512 --num_epochs 50 --scheduler_step_size 40 --learning_rate 1e-5 --gamma 0.5 --data_path ${season_train_data_path} --val_data_path ${season_val_data_path} --log_dir season_logs --model_name season_models  --frame_ids 0 -1 1 --batch_size 12 --val_batch_size 64 --pred_depth_scale 5.4 --scales 0 2 3 4 --num_workers 12 --load_weights_folder ./kitti_logs/kitti_ms_models/models/weights_19
```


#### generate predictions of seasondepth [test set](http://seasondepth-challenge.org/index/static/dataset/ICRA2022_SeasonDepth_Test_RGB.zip)
```
CUDA_VISIBLE_DEVICES=0  python pred_season_depth.py --encoder "van" --size_encoder "small" --data_path ${test_data_path} --pred_depth_path ${pred_depth_path} --eval_split seasondepth --load_weights_folder "${model_weight}" --eval_stereo --num_workers 16 --batch_size 128 --eval_set test
```
The test results of our model (baidunetdisk: https://pan.baidu.com/s/1cXoq1txyoIB6r-itXpFL5A code: 11uj) trained on kitti are as follows:
Mean AbsRel ↓	Mean a1 ↑	Variance AbsRel (10e-2) ↓	Variance a1 (10e-2) ↓	Relative Range AbsRel ↓	Relative Range a1 ↓
0.131	        0.852	      0.006	                  0.024	                      0.247	                  0.397
