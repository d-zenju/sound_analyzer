# coding: utf-8

import librosa
import librosa.display
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import csv


### MEMO
    # default: n_fft = 2048, win_length = n_fft,
    # hop_length = win_length / 4, D:np.ndarray[shape=(1+n_fft/2, t)]
    # frame = T * sr / hop_length

    # RMS = sqrt(1/n sum=(for i in d i^2))
    # RMS(db) = 20*log(RMS(V))

def rms(data):
    sum2 = 0
    for d in data:
        sum2 += d * d
        result = 20* np.log10(np.sqrt(sum2 / len(data)))
    return result


def calc_fft(data):
    ffts = []
    for y in ys:
        N = len(y)
        hann = np.hanning(N)
        wdat = hann * y
        wfft = np.fft.fft(wdat)
        freq = np.fft.fftfreq(N, d=1.0/sr)
        ffts.append([freq, wfft])
    return ffts


def calc_rms(data):
    rmss = []
    for i in data:
        rmss.append(rms(i))
        print(rmss)
    return rmss


def calc_psd(data):
    psds = []
    for y in ys:
        freq, psd = signal.welch(y, sr)
        psds.append([freq, psd])
    return psds


def calc_stft(data):
    stfts = []
    for y in ys:
        speg = librosa.stft(y)
        stfts.append(speg)
    return stfts


def split_wav(row, sr, offset):
    # Sampling Rate (Hz) : 22050
    beep = row[0+offset:11025+offset]
    nosound = row[12159+offset:41923+offset]
    eng_m = row[44100+offset:108135+offset]
    eng_f = row[121157+offset:184526+offset]
    jpn_f = row[198487+offset:224557+offset]
    jpn_m = row[236195+offset:286422+offset]
    google = row[294637+offset:325084+offset]
    dat = [row[0+offset:325084+offset], beep, nosound, eng_m, eng_f, jpn_f, jpn_m, google]
    return dat


def plt_wav(data, sr, title):
    titles = [
        'Sample', '440Hz Beep', 'No Sound', 'English Male', 'English Female',
        'Japanese Female', 'Japanese Male', 'OK, Google']
    xlabel = 'Samples'
    ylabel = 'Amplitude'
    plt.figure(figsize = (16, 9))
    for i in range(0,len(titles)):
        y = data[i]
        plt.subplot(2, 4, i + 1)
        plt.title(titles[i])
        librosa.display.waveplot(y, sr=sr)
    plt.tight_layout()
    plt.savefig('./fig/fig_wav/' + title + '.png')


def plt_fft(data, title, flag):
    titles = [
        'Sample', '440Hz Beep', 'No Sound', 'English Male', 'English Female',
        'Japanese Female', 'Japanese Male', 'OK, Google']
    plt.figure(figsize = (16, 9))
    plt.suptitle(title)
    for i in range(0,len(titles)):
        freq = data[i][0]
        amp = np.abs(data[i][1])
        n = len(freq)
        plt.subplot(2, 4, i + 1)
        plt.plot(freq[1:int(n/2)], amp[1:int(n/2)])
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.title(titles[i])
    plt.tight_layout()

    if flag == 0:
        # FFT
        plt.savefig('./fig/fig_fft/' + title + '.png')
    elif flag == 1:
        # PSD
        plt.savefig('./fig/fig_psd/' + title + '.png')


def plt_stft(data, title):
    titles = [
        'Sample', '440Hz Beep', 'No Sound', 'English Male', 'English Female',
        'Japanese Female', 'Japanese Male', 'OK, Google']
    plt.figure(figsize = (16, 9))
    plt.suptitle(title)
    for i in range(0,len(titles)):
        D = data[i]
        plt.subplot(2, 4, i + 1)
        librosa.display.specshow(
            librosa.amplitude_to_db(librosa.magphase(D)[0]),
            y_axis='log', x_axis='time')
        plt.title(titles[i])
        plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig('./fig/fig_stft/' + title + '.png')




# 音声ファイル読み込み
wav_fname = './180609_101823_113348.wav'
rowy, sr = librosa.load(wav_fname)
print ('Data Length : ' + str(len(rowy)))
print('Sampling Rate : ' + str(sr) + '[Hz]')

# 切り取り情報読み込み
csv_fname = './cut.csv'
with open(csv_fname, 'r') as cf:
    reader = csv.reader(cf)
    for i, st in enumerate(reader):

        if i < 230:
            continue

        print(i, st[0])

        # 音声ファイル分割
        print('分割')
        ys = split_wav(rowy, sr, int(st[0]))

        print(len(ys[0]))

        # RMS
        print('RMS')
        rmss =  calc_rms(ys)
        # FFT
        print('FFT')
        ffts = calc_fft(ys)
        # PSD
        print('PSD')
        psds = calc_psd(ys)
        # STFT
        print('STFT')
        stfts = calc_stft(ys)


        # SAVE
        np.save('./npy/npy_wav/npy_wav_' + str(i) + '.npy', ys)
        np.save('./npy/npy_rms/npy_rms_' + str(i) + '.npy', rmss)
        np.save('./npy/npy_fft/npy_fft_' + str(i) + '.npy', ffts)
        np.save('./npy/npy_psd/npy_psd_' + str(i) + '.npy', psds)

        plt_wav(ys, sr, 'Sound_Wave(' + str(i) + ')')
        plt_fft(ffts, 'FFT(' + str(i) + ')', 0)
        plt_fft(psds, 'PSD(' + str(i) + ')', 1)
        plt_stft(stfts, 'Power_spectrogram(' + str(i) + ')')
#plt.show()