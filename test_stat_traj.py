import numpy as np
import matplotlib.pyplot as plt


np_traj_stat = np.load("/cluster/scratch/arsood/save_val_result/save_sample_traj.npy", allow_pickle=True)
np_traj_stat_2 = np.load("/cluster/scratch/arsood/save_val_result/save_sample_traj_2.npy", allow_pickle=True)
np_traj_first_batch = np_traj_stat[1][2]
np_traj_first_batch_2 = np_traj_stat_2[1][2]
print(np_traj_first_batch[0]-np_traj_first_batch_2[0])
plt.figure()
plt.xlim(-20, 20)
traj_el = np.array(np_traj_first_batch[0])
traj_el_2 = np.array(np_traj_first_batch[1])
plt.plot(traj_el[...,1], traj_el[...,0], color= 'red')
plt.plot(traj_el_2[...,1], traj_el_2[...,0], color= 'green')
for index,traj in enumerate(np_traj_first_batch):
    if(index != 0):
        traj_el = np.array(traj)
        plt.plot(traj[...,1], traj[...,0], color = 'blue')
plt.savefig("/cluster/home/arsood/Semester_Project_Official/z_scratch_fig/image_sample.png")

