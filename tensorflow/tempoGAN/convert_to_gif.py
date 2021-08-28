import os
import time
load_model_test = 769  
load_model_no = 22
test_no = 40

output_path = '/nfs/hsu/repo/tempoGAN_master/tensorflow/2ddata_gan/test_%04d/out_%04d-%04d_%04d' % (load_model_test, load_model_test, load_model_no, test_no)
print(output_path)
os.system('cd ' + output_path + '/particles')
# convert particles
os.system('rm -rf list')
os.system("find . -type f -name '*.bmp' | sort -V > list")
os.system("mencoder mf://@list -mf fps=240:type=bmp -o ../particles.mp4 -ovc lavc")
os.system('cd ' + output_path)
os.system('ffmpeg -i particles.mp4 particles.gif')