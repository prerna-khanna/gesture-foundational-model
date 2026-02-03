import numpy as np
import matplotlib.pyplot as plt


activity = np.load('dataset/hhar/data_20_120.npy')
activity = activity[1,:20,:]

gesture = np.load('dataset/blind_user/data_20_120.npy')
gesture = gesture[1,:,:]

time_act = np.arange(0, len(activity)/20, 1/20)
time_ges = np.arange(0, len(gesture)/20, 1/20)

# Create a 2 rows, 1 column subplot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

# Plot the data
ax1.plot(time_act, activity, 'b-')
ax1.set_ylabel('Magnitude')
ax1.set_title('Signal 1')
ax1.grid(True, alpha=0.3)

ax2.plot(time_ges, gesture, 'r-')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Magnitude')
ax2.set_title('Signal 2')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('paper_figs/composite_human.png', dpi=300)