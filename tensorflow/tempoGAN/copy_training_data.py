import os

from numpy.core.numeric import full

res = 256
start_sim_no = 9000
start_dest_folder = 9000
n_sims = 30
last_frame = 100


use_wa = False
with_extra = False
with_ppos = True
folder = ''
if use_wa:
    folder = 'training_data_downsampled/wa/'
else:
    folder = 'training_data_downsampled/p2g/'

if with_extra:
    folder += 'with_extra'
else:
    folder += 'no_extra'

copy_cmd ='cp '
copy_from_prev = '/nfs/hsu/repo/MPM/mpm/output-2d-'
copy_to_prev = '../2ddata_sim/sim_'

command_pre = 'cp /nfs/hsu/repo/MPM/mpm/output-2d-'
command_mid = '-256x256/' + folder + '/* ../2ddata_sim/sim_'
command_post = '/'

cur_dest_folder = start_dest_folder
for sim in range(n_sims):
    cur_sim_no = start_sim_no + sim

    copy_from_fp = copy_from_prev + '%d-%dx%d' % (cur_sim_no, res, res)
    copy_from_fp_sub = copy_from_fp + '/%d' % (100)
    sim_complete = os.path.exists(copy_from_fp_sub)
    if not sim_complete:
        print('skipping %d'%cur_sim_no)
        continue

    print('copying {} to {}'.format(cur_sim_no, cur_dest_folder))
    full_command = command_pre + str(cur_sim_no) +command_mid + str(cur_dest_folder) + command_post
    os.system(full_command)
    if with_ppos:
        print('copy ppos')
        full_command = full_command.replace(folder, 'particle_positions')
        print(full_command)
        os.system(full_command)

    cur_dest_folder += 1
