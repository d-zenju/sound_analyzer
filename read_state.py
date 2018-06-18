import pandas as pd
import numpy as np
import sys
import pymap3d as pm
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def loadAWS1Data(fname, tstart=0, tend=sys.float_info.max):
    df = pd.read_csv(fname, index_col=0)
    df = df[(df.index > tstart) & (df.index < tend)]
    df = calcElapsedTime(df, basetime=tstart)
    df = df.reset_index().set_index('elapsed')
    return(df)


def calcElapsedTime(df, basetime=0):
    tn = df.index
    elapsed = []
    for t in tn:
        elapsed.append(float(t - basetime) / 10000000.0)
    df['elapsed'] = elapsed
    return df


def calcGeodesics(df, lat0, lon0, alt0):
    latn = df.loc[:,' lat']
    lonn = df.loc[:,' lon']
    altn = df.loc[:,' alt']

    azn = []
    eln = []
    dstn = []
    for i in range(len(latn)):
        az, el, dst = pm.geodetic2aer(latn.iat[i], lonn.iat[i], altn.iat[i], lat0, lon0, alt0)
        azn.append(az)
        eln.append(el)
        dstn.append(dst)
    df['azimuth'] = azn
    df['elevation'] = eln
    df['slantrange'] = dstn
    return df


'''
State DATA

fname = './state_15285067310000000.txt'
df = loadAWS1Data(fname, tstart=15285071030000000, tend=15285116280000000)
df = calcGeodesics(df, 35.61677460, 139.89745759, 6.57149905)
df.to_csv('./state_geodesics.csv')
'''


'''
RMS DATA

npy_rms = './npy/npy_rms/npy_rms_'
csv_fname = './cut.csv'
cut = pd.read_csv(csv_fname, header=None)
rms = []
for i in range(len(cut)):
    t = float(cut.iloc[i, 0]) / 22050.0
    nprms = np.load(npy_rms + str(i) + '.npy')
    lsrms = nprms.tolist()
    lsrms.insert(0, t)
    rms.append(lsrms)
dfrms = pd.DataFrame(rms, columns=['t', 'all', 'beep', 'nosound', 'eng_m', 'eng_f', 'jpn_f', 'jpn_m', 'google'])
dfrms.to_csv('./rms.csv')
'''