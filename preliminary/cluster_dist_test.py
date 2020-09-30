from main import *

csv_files = ['LL1-4_100012-F-AX1-8-42430-4-S129_pose_est_v2.csv', 'LL4-4_B6N_ControlMale-7-PSY_pose_est_v2.csv', 'LL6-2_B6N_Male_S6889224_ep2-BAT_pose_est_v2.csv', 'WT001G15N5F100227F-27-PSY_pose_est_v2.csv', 'LL4-3_B6SJLF1_F_pose_est_v2.csv']
vid_files = ['LL1-4_100012-F-AX1-8-42430-4-S129.avi', 'WT001G15N5F100227F-27-PSY.avi', 'LL4-4_B6N_ControlMale-7-PSY.avi', 'LL4-3_B6SJLF1_F.avi', 'LL6-2_B6N_Male_S6889224_ep2-BAT.avi']

csv_dir = '../../data/test/'
vid_dir = csv_dir + 'videos/'
for i, f in enumerate(csv_files):
    csv_files[i] = csv_dir + f
    vid_files[i] = vid_dir + vid_files[i]

labels = []
for i in range(len(csv_files)):
    labels.append(results(csv_files[i], vid_files[i], extract_frames=True))

with open('test_labels.sav', 'wb') as f:
    joblib.dump(labels, f)