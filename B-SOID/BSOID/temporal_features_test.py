import numpy as np
from BSOID.preprocessing import windowed_feats

def window_extracted_feats(feats, stride_window, temporal_dims=None):
    win_feats = []
    for f in feats:
        # indices 0-6 are link lengths, during windowing they should be averaged
        win_feats_ll = windowed_feats(f[:,:7], stride_window, mode='mean')
        
        # indices 7-13 are relative angles, during windowing they should be summed
        win_feats_rth = windowed_feats(f[:,7:13], stride_window, mode='sum')
def fft_over_window(feats, window_len):
    fft_feats = []
    N = feats.shape[0]

    for i in range(window_len, N, window_len):
        win_feats = feats[i-window_len:i,:]
        win_fft = np.fft.rfftn(win_feats, axes=[0])
        win_fft = win_fft.real ** 2 + win_fft.imag ** 2
        fft_feats.append(win_fft.reshape(1, -1))

    fft_feats = np.vstack(fft_feats)

    return fft_feats