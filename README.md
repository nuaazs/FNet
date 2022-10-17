
## 222-10-14更新

[x] 第一步：根据4DCT文件获取DDF矩阵（UNet,F02服务器上）
[x] 第二步：根据DDF和T0时项的3DCT生成s4DCT文件（目的是，保证DDF矩阵为标准答案）
[x] 第三步：DDF loss + image loss 训练 FNet 
[ ] 第四步：根据保存的中间结果（npy文件），计算肺部体积变化曲线
[ ] 第五步：根据保存的中间结果（npy文件），可视化横膈膜偏移量
[ ] 第六步：利用3Dslicer完成肿瘤靶区勾画
[ ] 第七步：根据保存的DDF结果（npy文件），计算肿瘤部位的平均值差异。

## B. Train
```shell
python main.py --name fnet
```

## C. Postprocess

### Step 1:
```shell
python test.py
```

### Step 2:
get input image(`nii.gz`): change `.npy` (output from test.py) -> `nii.gz` image(save to ./output_niis)
```shell
python get_nii_by_ddf.py
```

### Step 3:
get lung volume change.
```shell
cd postprocess
python ./check_results.py
```


### Step 4:
get tumor motion change.