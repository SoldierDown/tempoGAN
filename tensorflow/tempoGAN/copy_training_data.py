import os


start_sim_no = 1018
start_dest_folder = 7777
n_sims = 1

command_pre = 'cp /nfs/hsu/repo/MPM/mpm/output-2d-'
command_mid = '-256x256/training_data_downsampled/* ../2ddata_sim/sim_'
command_post = '/'

for sim in range(n_sims):
    cur_sim_no = start_sim_no + sim
    cur_dest_folder = start_dest_folder + sim
    full_command = command_pre + str(cur_sim_no) +command_mid + str(cur_dest_folder) + command_post
    os.system(full_command)