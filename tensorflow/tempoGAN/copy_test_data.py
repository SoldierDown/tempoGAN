import os

from numpy.core.numeric import full

res = 64
start_sim_no = 9000
start_dest_folder = 9999
n_sims = 1
last_frame = 2


use_wa = False
with_extra = False
with_ppos = True
folder = 'test_data_coarse_fine'

copy_cmd ='cp '
copy_from_prev = '/nfs/hsu/repo/MPM/mpm/output-2d-'
copy_to_prev = '../2ddata_sim/sim_'

command_pre = 'cp /nfs/hsu/repo/MPM/mpm/output-2d-'
command_mid = '-64x64/' + folder + '/* ../2ddata_sim/sim_'
command_post = '/'

cur_dest_folder = start_dest_folder
for sim in range(n_sims):
    cur_sim_no = start_sim_no + sim
    print('copying {} to {}'.format(cur_sim_no, cur_dest_folder))
    full_command = command_pre + str(cur_sim_no) +command_mid + str(cur_dest_folder) + command_post
    cur_dest_folder += 1
    os.system(full_command)
    if with_ppos:
        print('copy ppos')
        full_command = full_command.replace(folder, 'particle_positions')
        print(full_command)
        os.system(full_command)

    cur_dest_folder += 1
