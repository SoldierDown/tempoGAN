# Modification
Use anaconda to manage python packages
## Dependency
sudo apt-get install python-dev
## Python include/lib path
in tempoGAN/CMakeLists.txt
```
set(NUMPY_INCLUDE_DIR "/home/ubuntu/anaconda3/envs/tempoGAN/lib/python3.7/site-packages/numpy/core/include")
...
...
set(PYTHON_INCLUDE_DIRS "/home/ubuntu/anaconda3/envs/tempoGAN_otf/include/python3.7")
set(PYTHON_LIBRARIES "/home/ubuntu/anaconda3/envs/tempoGAN_otf/lib/libpython3.7m.so")
```
## Package version
1. in tempoGAN/tensorflow/tools/GAN.py, change 
```
from keras import backend as kb
```
to 
```
from tensorflow.keras import backend as kb
```
2. in tempoGAN/tensorflow/tempoGAN/tempoGAN.py, change
```
# tf.set_random_seed(randSeed)
```
to
```
tf.random.set_seed(randSeed)
```
## Warning
in tempoGAN/source/pwrapper/numpyWrap.cpp
change 
PyMODINIT_FUNC initNumpy() { import_array(); }
to 
PyMODINIT_FUNC initNumpy() { import_array(); return NULL;}

# Compile (/tempoGAN/build)
## with GUI
cmake .. -DGUI=ON -DOPENMP=ON -DNUMPY=ON
## without GUI
cmake .. -DGUI=OFF -DOPENMP=ON -DNUMPY=ON

# Run (tempoGAN/tensorflow/tempoGAN)
## Generate training data
../../build/manta ../datagen/gen_sim_data.py basePath ../2ddata_sim/ reset 1 saveuni 1 gui 0
## Training
python example_run_training.py
## Generate new data
python example_run_output.py
# AWS + git + X forwarding
ssh -X -A hsu-aws

# undo git add
git rm -r  --cached  <files>