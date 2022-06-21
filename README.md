# Exploring CUDA Bank Conflict [![Windows](https://github.com/Ahdhn/BankConflict/actions/workflows/Windows.yml/badge.svg)](https://github.com/Ahdhn/BankConflict/actions/workflows/Windows.yml) [![Ubuntu](https://github.com/Ahdhn/BankConflict/actions/workflows/Ubuntu.yml/badge.svg)](https://github.com/Ahdhn/BankConflict/actions/workflows/Ubuntu.yml)

# Experiments 
In all experiments, we used `num_repeat=1` i.e., there is only one read from shared memory that may suffer from bank conflicts. All other writes to shared memory are bank-conflict free. All writes to global memory are coalesced. The `offset` parameter defines the number of bank conflicts (plus one) i.e., `offset= 1` is bank conflict free. 

## RTX A6000
Size = 2^20 of `uint32_t`

``` 
offset= 1 , time(ms)= 0.211904
offset= 2 , time(ms)= 0.324608
offset= 3 , time(ms)= 0.439296
offset= 4 , time(ms)= 0.555008
offset= 5 , time(ms)= 0.67072
offset= 6 , time(ms)= 0.784384
offset= 7 , time(ms)= 0.900096
offset= 8 , time(ms)= 1.01376
offset= 9 , time(ms)= 1.1305
offset= 10 , time(ms)= 1.24518
offset= 11 , time(ms)= 1.77459
offset= 12 , time(ms)= 1.49402
offset= 13 , time(ms)= 1.59846
offset= 14 , time(ms)= 1.71517
offset= 15 , time(ms)= 3.48576
offset= 16 , time(ms)= 1.96403
offset= 17 , time(ms)= 2.08691
offset= 18 , time(ms)= 2.2057
offset= 19 , time(ms)= 3.08019
offset= 20 , time(ms)= 3.20307
offset= 21 , time(ms)= 2.57434
offset= 22 , time(ms)= 2.68902
offset= 23 , time(ms)= 4.39501
offset= 24 , time(ms)= 2.90816
offset= 25 , time(ms)= 3.08736
offset= 26 , time(ms)= 4.5312
offset= 27 , time(ms)= 5.06067
offset= 28 , time(ms)= 3.41709
offset= 29 , time(ms)= 6.04381
offset= 30 , time(ms)= 3.65875
offset= 31 , time(ms)= 3.76845
offset= 32 , time(ms)= 5.67706
``` 

## Quadro GV100
Size = 2^20 of `uint32_t`
```
offset= 1 , time(ms)= 0.862336
offset= 2 , time(ms)= 0.896128
offset= 3 , time(ms)= 0.972832
offset= 4 , time(ms)= 1.09117
offset= 5 , time(ms)= 1.24666
offset= 6 , time(ms)= 1.40691
offset= 7 , time(ms)= 1.54125
offset= 8 , time(ms)= 1.71981
offset= 9 , time(ms)= 1.91075
offset= 10 , time(ms)= 2.07248
offset= 11 , time(ms)= 2.28746
offset= 12 , time(ms)= 2.45683
offset= 13 , time(ms)= 2.67293
offset= 14 , time(ms)= 2.87142
offset= 15 , time(ms)= 3.13786
offset= 16 , time(ms)= 3.26598
offset= 17 , time(ms)= 3.39021
offset= 18 , time(ms)= 3.56384
offset= 19 , time(ms)= 3.6984
offset= 20 , time(ms)= 3.90771
offset= 21 , time(ms)= 4.07485
offset= 22 , time(ms)= 4.27043
offset= 23 , time(ms)= 4.42349
offset= 24 , time(ms)= 4.63091
offset= 25 , time(ms)= 4.92314
offset= 26 , time(ms)= 5.19203
offset= 27 , time(ms)= 5.42634
offset= 28 , time(ms)= 5.49341
offset= 29 , time(ms)= 5.79174
offset= 30 , time(ms)= 5.99962
offset= 31 , time(ms)= 6.16326
offset= 32 , time(ms)= 6.34682
```


## Build 

```
mkdir build
cd build 
cmake ..
```

Depending on the system, this will generate either a `.sln` project on Windows or a `make` file for a Linux system. 
