import os

axis = 1
start_dest_folder = 9100 + axis * 100
n_sims = 30

command_pre = "rename 's/velocity/density/' ../2ddata_sim/sim_"
command_post = '/*'
for sim in range(n_sims):
    cur_dest_folder = start_dest_folder + sim
    print('renaming {}'.format(cur_dest_folder))
    full_command = command_pre + str(cur_dest_folder) + command_post
    os.system(full_command)