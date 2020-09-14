import ffmpeg
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def create_confusion_matrix(feats, labels, clf):
    pred = clf.predict(feats)
    data = confusion_matrix(labels, pred, normalize='all')
    df_cm = pd.DataFrame(data, columns=np.unique(pred), index=np.unique(pred))
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, cmap="Blues", annot=False)
    plt.show()

    return data

def frames_from_video(video_file, frame_dir):
    probe = ffmpeg.probe(video_file)
    video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    width = int(video_info['width'])
    height = int(video_info['height'])
    num_frames = int(video_info['nb_frames'])
    bit_rate = int(video_info['bit_rate'])
    avg_frame_rate = round(int(video_info['avg_frame_rate'].rpartition('/')[0]) / int(video_info['avg_frame_rate'].rpartition('/')[2]))
    logging.info('frame extraction for {} frames at {} frames per second'.format(num_frames, avg_frame_rate))
    try:
        (ffmpeg.input(video_file)
            .filter('fps', fps=avg_frame_rate)
            .output(str.join('', (frame_dir, '/frame%01d.png')), video_bitrate=bit_rate,
                    s=str.join('', (str(int(width * 0.5)), 'x', str(int(height * 0.5)))), sws_flags='bilinear',
                    start_number=0)
            .run(capture_stdout=True, capture_stderr=True))
    except ffmpeg.Error as e:
        print('stdout:', e.stdout.decode('utf8'))
        print('stderr:', e.stderr.decode('utf8'))