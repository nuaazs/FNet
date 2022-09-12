# Train
```shell
python main.py --name fnet
```


# Post Processing

## Step 0:
```shell
python test.py
```

## Step 1:
get input image(`nii.gz`): change `.npy` (output from test.py) -> `nii.gz` image(save to ./output_niis)
```shell
python get_nii_by_ddf.py
```

## Step 2:
get lung volume change.

## Step 3:
get tumor motion change.