# Oliver-TimeSeriesForecast
Code, Demo and Results of the homework program for DL class in UPC.

## Note
This repo contains four comparative experiments for Multivariate time series forecasting problem.

## Datasets
Four datasets can be downloaded [here](https://github.com/laiguokun/multivariate-time-series-data).
Unzip the dataset in `./data/`.

## Models
I've conducted test on four different models: CNN, RNN, [MHA_Net](https://arxiv.org/abs/1706.03762), [LSTNet](https://arxiv.org/abs/1703.07015), which can be found in `./models/`.

## Demo
```
# in {path_to_this_repo}/,
$ bash All_test.sh
```
One can also customize own settings in shell script. Parameters are listed in `./args_file.py`.
## Results

16 Pretrained models (4 models * 4 datasets) are available in `./save_models`. 

More experimental details can be found in `./logs`, `./csv_files` and `evalution_pics`.
## Contact
Please contact me if there is any question. (Chao Wang oliversavealien@gmail.com)
