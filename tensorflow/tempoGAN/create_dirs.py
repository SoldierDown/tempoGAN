import os

axis = 1
start_dest_folder = 1000 + axis * 1000
n_sims = 30
start_dest_folder = 4000
command_pre = "mkdir ../2ddata_sim/sim_"
command_post = '/*'
for sim in range(n_sims):
    cur_dest_folder = start_dest_folder + sim
    print('create {}'.format(cur_dest_folder))
    full_command = command_pre + str(cur_dest_folder)
    os.system(full_command)