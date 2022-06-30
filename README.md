# Exploring CUDA Bank Conflict [![Windows](https://github.com/Ahdhn/BankConflict/actions/workflows/Windows.yml/badge.svg)](https://github.com/Ahdhn/BankConflict/actions/workflows/Windows.yml) [![Ubuntu](https://github.com/Ahdhn/BankConflict/actions/workflows/Ubuntu.yml/badge.svg)](https://github.com/Ahdhn/BankConflict/actions/workflows/Ubuntu.yml)

# Experiments 
In all experiments, we used `num_repeat=1` i.e., there is only one read from shared memory that may suffer from bank conflicts. All other writes to shared memory are bank-conflict free. All writes to global memory are coalesced. The `offset` parameter defines the number of bank conflicts (plus one) i.e., `offset= 1` is bank conflict free. 

## RTX A6000
Size = 2^20 of `uint32_t`. Average over 100 runs 

``` 
 offset= 1, time(ms)= 4.30209
 offset= 2, time(ms)= 4.34682
 offset= 3, time(ms)= 4.3636
 offset= 4, time(ms)= 4.33606
 offset= 5, time(ms)= 4.36726
 offset= 6, time(ms)= 4.36421
 offset= 7, time(ms)= 4.36835
 offset= 8, time(ms)= 4.41139
 offset= 9, time(ms)= 4.38105
 offset= 10, time(ms)= 4.37528
 offset= 11, time(ms)= 4.37208
 offset= 12, time(ms)= 4.37521
 offset= 13, time(ms)= 4.37823
 offset= 14, time(ms)= 4.33889
 offset= 15, time(ms)= 4.35368
 offset= 16, time(ms)= 4.36441
 offset= 17, time(ms)= 4.34809
 offset= 18, time(ms)= 4.38047
 offset= 19, time(ms)= 4.3697
 offset= 20, time(ms)= 4.35399
 offset= 21, time(ms)= 4.39009
 offset= 22, time(ms)= 4.48932
 offset= 23, time(ms)= 4.61049
 offset= 24, time(ms)= 4.56469
 offset= 25, time(ms)= 4.48154
 offset= 26, time(ms)= 4.65752
 offset= 27, time(ms)= 4.57338
 offset= 28, time(ms)= 4.38176
 offset= 29, time(ms)= 4.35502
 offset= 30, time(ms)= 4.38195
 offset= 31, time(ms)= 4.4514
 offset= 32, time(ms)= 4.41471
``` 

## Quadro GV100
Size = 2^20 of `uint32_t`. Average over 100 runs 
```
 offset= 1, time(ms)= 5.46449
 offset= 2, time(ms)= 5.46254
 offset= 3, time(ms)= 5.45825
 offset= 4, time(ms)= 5.46375
 offset= 5, time(ms)= 5.46334
 offset= 6, time(ms)= 5.46545
 offset= 7, time(ms)= 5.462
 offset= 8, time(ms)= 5.46419
 offset= 9, time(ms)= 5.46712
 offset= 10, time(ms)= 5.46389
 offset= 11, time(ms)= 5.46584
 offset= 12, time(ms)= 5.46124
 offset= 13, time(ms)= 5.46077
 offset= 14, time(ms)= 5.46143
 offset= 15, time(ms)= 5.46422
 offset= 16, time(ms)= 5.46826
 offset= 17, time(ms)= 5.46477
 offset= 18, time(ms)= 5.46525
 offset= 19, time(ms)= 5.463
 offset= 20, time(ms)= 5.45978
 offset= 21, time(ms)= 5.4587
 offset= 22, time(ms)= 5.46318
 offset= 23, time(ms)= 5.46549
 offset= 24, time(ms)= 5.46769
 offset= 25, time(ms)= 5.46521
 offset= 26, time(ms)= 5.46732
 offset= 27, time(ms)= 5.4607
 offset= 28, time(ms)= 5.459
 offset= 29, time(ms)= 5.46325
 offset= 30, time(ms)= 5.46344
 offset= 31, time(ms)= 5.46321
 offset= 32, time(ms)= 5.46987
```

## Build 

```
mkdir build
cd build 
cmake ..
```

Depending on the system, this will generate either a `.sln` project on Windows or a `make` file for a Linux system. 
