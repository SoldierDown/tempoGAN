# Trial Record
1. aug + tempoD                         works: 38 37
2. aug, no tempoD                       
3. no aug, tempoD
4. no aug, no tempoD                    works: 39 39, and both 64x64 and 128x128 input works

case 2024: v = (0, 5)



case 2124: v = (0, 0.1)
output model: 57-13

model 77: train on case 2123 v = (0 5)



relu works for all 1s 64x64 to all 1s 256x256: 80-799
direct output without activation functions works for all ones 64x64 to all ones 256x256: 89-587
should also works for all 5s 64x64 to all 5s 256x256, without activation functions: 90-799, need to iterate more: mean_v: [3.6194744], max_v: [5.7436705], mean_v: [4.9534063]

works for all -5s 64x64 to all -5 256x256, without activation functions: 95-779


only generator
works for all 1s 64x64 to all 1 256x256, without activation functions: 130-1824
works for all -5s 64x64 to all -5 256x256, without activation functions: 132-1263

148/149-326 best so far with lr = 1e-4 and 1e-5 respectively
152-556 with lr = 1e-6

get better using data augmentation
153-643 with lr = 1e-4, pos_mean_v = [4.81]

Training
sim_9600 - sim_9017: training datasets with density_low_*, velocity_low_*, velocity_high_*
Test
sim_9999


use density, no Ds, no Dt, frames 10:   403, 1043
use density, w. Ds, no Dt, frames 10:  




# Training Data
1. sim_1000 ~ sim_1023: p2g
2. sim_2000 ~ sim_2023: wa
