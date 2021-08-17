import os

axis = 0
start_dest_folder = 3000 + axis * 100
n_sims = 60

command_pre = 'rm ../2ddata_sim/sim_'
command_post = '/*'
for sim in range(n_sims):
    cur_dest_folder = start_dest_folder + sim
    print('removing {}'.format(cur_dest_folder))
    full_command = command_pre + str(cur_dest_folder) + command_post
    os.system(full_command)