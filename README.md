## PyNET-QxQ


## Setups
### Requirements
See `requirements.txt`.
     
### Data
- Download [3CCD RAW and QxQ dataset](https://drive.google.com/file/d/1_aQqoldM1rGR9nhz6tiaYMqpAFKjYbtu/view?usp=sharing) and extract it.
(These images are stored in 16-bit raw format. A third-party image viewer is required to handle 16-bit `RAW` images.)
  
`PNG` images should be located in `./raw_images/test/`.
The `test` folder consists of `msc` (input images) and `gt` (target output images).
To compute validation metrics, you must exclude QxQ images from test (since QxQ images do not have ground truth images).

### Parameters
You should put the parameter file in the `./Experiments/models` folder with the name `best_acc_model.pth`.

## Run models
1) Run in cpu-only mode:
```
python main.py --cpu
```

2) Run with cuda:
```
python main.py
```
Test outputs will be stored in `./Experiments/test_images`.

## Note
You can update options in `option.py` to modify settings.
For example, to output validation metrics, you can activate `--metric` option. 
```
python main.py --metric
```


