# Exploring CUDA Bank Conflict [![Windows](https://github.com/Ahdhn/BankConflict/actions/workflows/Windows.yml/badge.svg)](https://github.com/Ahdhn/BankConflict/actions/workflows/Windows.yml) [![Ubuntu](https://github.com/Ahdhn/BankConflict/actions/workflows/Ubuntu.yml/badge.svg)](https://github.com/Ahdhn/BankConflict/actions/workflows/Ubuntu.yml)

# Experiments 
In all experiments, we used `num_repeat=1` i.e., there is only one read from shared memory that may suffer from bank conflicts. All other writes to shared memory are bank-conflict free. All writes to global memory are coalesced. The `offset` parameter defines the number of bank conflicts (plus one) i.e., `offset= 1` is bank conflict free. 

## RTX A6000
Size = 2^30 of `uint32_t`. Average over 100 runs 

``` 
 offset= 1, time(ms)= 44.143
 offset= 2, time(ms)= 47.1548
 offset= 3, time(ms)= 48.5601
 offset= 4, time(ms)= 46.6286
 offset= 5, time(ms)= 44.3103
 offset= 6, time(ms)= 44.3222
 offset= 7, time(ms)= 44.2231
 offset= 8, time(ms)= 45.6252
 offset= 9, time(ms)= 44.2273
 offset= 10, time(ms)= 44.3881
 offset= 11, time(ms)= 44.2617
 offset= 12, time(ms)= 44.5928
 offset= 13, time(ms)= 44.2277
 offset= 14, time(ms)= 44.4022
 offset= 15, time(ms)= 44.2765
 offset= 16, time(ms)= 49.6134
 offset= 17, time(ms)= 44.2303
 offset= 18, time(ms)= 44.375
 offset= 19, time(ms)= 44.2365
 offset= 20, time(ms)= 44.4856
 offset= 21, time(ms)= 44.2566
 offset= 22, time(ms)= 44.3398
 offset= 23, time(ms)= 44.231
 offset= 24, time(ms)= 45.6863
 offset= 25, time(ms)= 44.2759
 offset= 26, time(ms)= 44.3077
 offset= 27, time(ms)= 44.317
 offset= 28, time(ms)= 44.4886
 offset= 29, time(ms)= 44.2222
 offset= 30, time(ms)= 44.3496
 offset= 31, time(ms)= 44.2922
 offset= 32, time(ms)= 59.4622
 offset= 33, time(ms)= 44.213
 offset= 34, time(ms)= 44.3181
 offset= 35, time(ms)= 44.3126
``` 

## Quadro GV100
Size = 2^30 of `uint32_t`. Average over 100 runs 
```

```

## Build 

```
mkdir build
cd build 
cmake ..
```

Depending on the system, this will generate either a `.sln` project on Windows or a `make` file for a Linux system. 
