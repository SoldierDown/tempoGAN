import os

start_dest_folder = 6666
n_sims = 1

command_pre = 'rm ../2ddata_sim/sim_'
command_post = '/*'
for sim in range(n_sims):
    cur_dest_folder = start_dest_folder + sim
    full_command = command_pre + str(cur_dest_folder) + command_post
    os.system(full_command)