# van_depth
visual attention network based monocular depth estimation

#### 1. install
```
conda create -n van_depth python=3.7 -y 
conda activate van_depth
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -y #pytorch1.11.0
pip install opencv-python==4.5.2.52 tensorboardX==1.5 scikit-image==0.16.2 timm==0.4.12
pip install tqdm==4.57.0 xlrd==2.0.1 xlwt==1.3.0
```

#### 2. prepare kitti dataset
Following [monodepth2](https://github.com/nianticlabs/monodepth2#-kitti-training-data)

#### 3. download pretrained [VAN](https://github.com/Visual-Attention-Network/VAN-Classification)
```
mkdir pretrained
wget -P ./pretrained/ https://huggingface.co/Visual-Attention-Network/VAN-Small-original/resolve/main/van_small_811.pth.tar
```

#### 4. train on kitti
```
CUDA_VISIBLE_DEVICES=0 python kitti_train.py --data_path ${kitti_data_path} --log_dir ./kitti_logs --model_name "kitti_ms_models"  --frame_ids 0 -1 1 --scales 0 2 3 4 --use_stereo
```

#### 5. test on the eigen split of kitti
```
python export_kitti_gt_depth.py --data_path ${data_path} --split eigen #export gt depth
CUDA_VISIBLE_DEVICES=0 python evaluate_kitti_depth.py --scales 0 2 3 4 --data_path ${kitti_data_path} --load_weights_folder "${model_weight}" --eval_stereo
```
The test results of our [model](https://pan.baidu.com/s/14ohVPXOnWKj7krq4N7ycsA) (baidunetdisk extraction code: 5wh7) trained on kitti are as follows:
| abs_rel | sq_rel | rmse  | rmse_log | a1    | a2    | a3    |
|---------|--------|-------|----------|-------|-------|-------|
| 0.099   | 0.775  | 4.570 | 0.186    | 0.893 | 0.961 | 0.981 |

#### 6. prepare seasondepth dataset
Donload and unzip [train set](https://doi.org/10.6084/m9.figshare.16442025) ans [val set](https://doi.org/10.6084/m9.figshare.14731323)

Prepare [test set](http://seasondepth-challenge.org/index/static/dataset/ICRA2022_SeasonDepth_Test_RGB.zip) following [seasondepth](https://github.com/SeasonDepth/SeasonDepth/tree/master/dataset_info)

#### 7. fine-tune on seasondepth
```
CUDA_VISIBLE_DEVICES=6 python season_train.py --set_seed --split seasondepth --height 384 --width 512 --num_epochs 50 --scheduler_step_size 40 --learning_rate 1e-5 --gamma 0.5 --data_path ${season_train_data_path} --val_data_path ${season_val_data_path} --log_dir season_logs --model_name season_models  --frame_ids 0 -1 1 --batch_size 12 --val_batch_size 64 --pred_depth_scale 5.4 --scales 0 2 3 4 --num_workers 12 --load_weights_folder ./kitti_logs/kitti_ms_models/models/weights_19
```


#### 8. generate depth predictions of seasondepth [test set](http://seasondepth-challenge.org/index/static/dataset/ICRA2022_SeasonDepth_Test_RGB.zip)
```
CUDA_VISIBLE_DEVICES=0  python pred_season_depth.py --encoder "van" --size_encoder "small" --data_path ${season_test_data_path} --pred_depth_path "${model_weight}/van_depth_test_predictions/slice2_3_7_8" --eval_split seasondepth --load_weights_folder "${model_weight}" --eval_stereo --num_workers 16 --batch_size 128 --eval_set test
```
The test results of our [model](https://pan.baidu.com/s/1cXoq1txyoIB6r-itXpFL5A) (baidunetdisk extraction code: 11uj) trained on seasondepth are as follows:

| Mean AbsRel ??? | Mean a1 ??? | Variance AbsRel (10e-2) ??? | Variance a1 (10e-2) ??? | Relative Range AbsRel ??? | Relative Range a1 ??? |
|---------------|-----------|---------------------------|-----------------------|-------------------------|---------------------|
| 0.131         | 0.852     | 0.006                     | 0.024                 | 0.247                   | 0.397               |


#### Acknowledgment
Our implementation is mainly based on [monodepth2](https://github.com/nianticlabs/monodepth2#-kitti-training-data) and [VAN](https://github.com/Visual-Attention-Network/VAN-Classification). Thanks for their authors.
