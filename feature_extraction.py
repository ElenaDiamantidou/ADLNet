"""
Project Name : ADLNet
author       : Eleni Diamantides
email        : elenadiamantidou@gmail.com
"""

import json
import sys, errno, os

import pandas as pd
import matplotlib.pyplot as plt
from math import acos, sqrt

import warnings
import librosa
import numpy as np
from numpy.fft import *

from scipy import stats, spatial
from scipy.stats import entropy, iqr
from scipy.signal import butter, lfilter, filtfilt, stft
from scipy import fftpack
from scipy.special import entr

warnings.filterwarnings("ignore")


def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def means(data, prefix, mag):
    df_mean = pd.DataFrame()
    if mag:
        df_mean[prefix] = data.mean(axis=1)
    else:
        for key in data.keys():
            df_mean[prefix + key] = data[key].mean(axis=1)
    return df_mean


def stds(data, prefix, mag):
    df_std = pd.DataFrame()
    if mag:
        df_std[prefix] = data.std(axis=1)
    else:
        for key in data.keys():
            df_std[prefix + key] = data[key].std(axis=1)
    return df_std


def mad(data, prefix, mag):
    df_mad = pd.DataFrame()
    if mag:
        df_mad[prefix] = data.mad(axis=1)
    else:
        for key in data.keys():
            df_mad[prefix + key] = data[key].mad(axis=1)
    return df_mad


def percentile(data, prefix, mag):
    df_percentile = pd.DataFrame()
    if mag:
        df_percentile[''.join([prefix.split('_')[0], '25', prefix.split('_')[1]])] = \
            np.percentile(data, 25, axis=1)
        df_percentile[''.join([prefix.split('_')[0], '50', prefix.split('_')[1]])] = \
            np.percentile(data, 50, axis=1)
        df_percentile[''.join([prefix.split('_')[0], '75', prefix.split('_')[1]])] = \
            np.percentile(data, 75, axis=1)
    else:
        for key in data.keys():
            df_percentile[''.join([prefix, '25', key])] = np.percentile(data[key], 25, axis=1)
            df_percentile[''.join([prefix, '50', key])] = np.percentile(data[key], 50, axis=1)
            df_percentile[''.join([prefix, '75', key])] = np.percentile(data[key], 75, axis=1)

    return df_percentile


def moments(data, prefix, mag):
    df_moment = pd.DataFrame()
    if mag:
        df_moment[''.join([prefix.split('_')[0], '3', prefix.split('_')[1]])] = \
            stats.moment(data, moment=3, axis=1)
        df_moment[''.join([prefix.split('_')[0], '4', prefix.split('_')[1]])] = \
            stats.moment(data, moment=4, axis=1)
    else:
        for key in data.keys():
            df_moment[''.join([prefix, '3', key])] = stats.moment(data[key], moment=3, axis=1)
            df_moment[''.join([prefix, '4', key])] = stats.moment(data[key], moment=4, axis=1)
    return df_moment


def min_max(data, prefix, mag):
    df_min_max = pd.DataFrame()
    if mag:
        df_min_max[prefix[0]] = data.min(axis=1)
        df_min_max[prefix[1]] = data.max(axis=1)
    else:
        for key in data.keys():
            df_min_max[prefix[0] + key] = data[key].min(axis=1)
        for key in data.keys():
            df_min_max[prefix[1] + key] = data[key].max(axis=1)
    return df_min_max


def correlation(data, prefix, sensors):
    df_corr = pd.DataFrame()
    for s in sensors:
        df_corr[prefix + s + 'XY'] = data[s + 'X'].corrwith(data[s + 'Y'], axis=1)
        df_corr[prefix + s + 'XZ'] = data[s + 'X'].corrwith(data[s + 'Z'], axis=1)
        df_corr[prefix + s + 'YZ'] = data[s + 'Y'].corrwith(data[s + 'Z'], axis=1)

    return df_corr


def cosine_distance(data, prefix, sensors):
    cos_values = []
    df_cos = pd.DataFrame()
    # Sensor Cosine Distance
    for s in sensors:
        for index, (x, y, z) in enumerate(zip(data[s + 'X'].values, data[s + 'Y'].values, data[s + 'Z'].values)):
            # X, Y, Z Measurements in the Window
            xy = 1 - spatial.distance.cosine(x, y)
            xz = 1 - spatial.distance.cosine(x, z)
            yz = 1 - spatial.distance.cosine(y, z)
            cos_values.append([xy, xz, yz])
        df_cos_sensor = pd.DataFrame(cos_values, columns=[prefix + s + 'XY', prefix + s + 'XZ', prefix + s + 'YZ'])
        df_cos = pd.concat([df_cos, df_cos_sensor], axis=1)

    return df_cos


def kurtosis(data, prefix, mag):
    df_kurtosis = pd.DataFrame()
    if mag:
        df_kurtosis[prefix] = data.kurtosis(axis=1)
    else:
        for key in data.keys():
            df_kurtosis[prefix + key] = data[key].kurtosis(axis=1)

    return df_kurtosis


def skewness(data, prefix, mag):
    df_skewness = pd.DataFrame()
    if mag:
        df_skewness[prefix] = data.skew(axis=1)
    else:
        for key in data.keys():
            df_skewness[prefix + key] = data[key].skew(axis=1)

    return df_skewness


def interquartile(data, prefix, mag):
    df_iqr = pd.DataFrame()
    if mag:
        df_iqr[prefix] = iqr(data, axis=1)
        df_iqr[prefix + 'Dev'] = iqr(data, axis=1) / 2
    else:
        for key in data.keys():
            df_iqr[prefix + key] = iqr(data[key], axis=1)
            df_iqr[prefix + 'Dev' + key] = iqr(data[key], axis=1) / 2
    return df_iqr


def autocorrelation(data, prefix):
    autocorrs = []
    for i, row in data.iterrows():
        # subtract the average magnitude
        row = row - row.mean()
        # calculate auto-correlation
        autocorrs.append(row.autocorr())

    autocorr = pd.DataFrame({prefix: autocorrs})

    return autocorr


def signal_entropy(data, prefix):
    df_entropy = pd.DataFrame()
    for key in data.keys():
        df_entropy[prefix + key] = entropy(abs(data[key]), axis=1)

    return df_entropy


def energy_axial(data, prefix):
    df_energies = pd.DataFrame()
    for key in data.keys():
        df_energies[prefix + key] = (data[key] ** 2).sum(axis=1)

    return df_energies


def sma(data, prefix):
    """
    signal magnitude area
    """
    df_sma = pd.DataFrame()
    for key in data.keys():
        df_sma[prefix + key] = abs(data[key]).sum(axis=1)
    return df_sma


def arburg2(data, order, prefix, mag):
    def ar(x):
        N = len(x)

        if order == 0.:
            raise ValueError("order must be > 0")

        # Initialisation
        # rho, den
        rho = sum(abs(x) ** 2.) / N  # Eq 8.21 [Marple]_
        den = rho * 2. * N

        # backward and forward errors
        ef = np.zeros(N, dtype=complex)
        eb = np.zeros(N, dtype=complex)
        for j in range(0, N):  # eq 8.11
            ef[j] = x[j]
            eb[j] = x[j]

        # AR order to be stored
        a = np.zeros(1, dtype=complex)
        a[0] = 1
        # rflection coeff to be stored
        ref = np.zeros(order, dtype=complex)

        E = np.zeros(order + 1)
        E[0] = rho

        for m in range(0, order):
            # Calculate the next order reflection (parcor) coefficient
            efp = ef[1:]
            ebp = eb[0:-1]
            num = -2. * np.dot(ebp.conj().transpose(), efp)
            den = np.dot(efp.conj().transpose(), efp)
            den += np.dot(ebp, ebp.conj().transpose())
            ref[m] = num / den

            # Update the forward and backward prediction errors
            ef = efp + ref[m] * ebp
            eb = ebp + ref[m].conj().transpose() * efp

            # Update the AR coeff.
            a.resize(len(a) + 1)
            a = a + ref[m] * np.flipud(a).conjugate()

            # Update the prediction error
            E[m + 1] = np.real((1 - ref[m].conj().transpose() * ref[m])) * E[m]
        return a

    df_arburg = pd.DataFrame()
    if mag:
        coeff1, coeff2, coeff3, coeff4 = [], [], [], []
        for i, row in data.iterrows():
            x = np.array(row)
            a = ar(x)
            coeff1.append(a[1:][0].real)
            coeff2.append(a[1:][1].real)
            coeff3.append(a[1:][2].real)
            coeff4.append(a[1:][3].real)

        df_arburg[prefix + 'Coeff1'] = coeff1
        df_arburg[prefix + 'Coeff2'] = coeff2
        df_arburg[prefix + 'Coeff3'] = coeff3
        df_arburg[prefix + 'Coeff4'] = coeff4

    else:
        for key in data.keys():
            coeff1, coeff2, coeff3, coeff4 = [], [], [], []
            for i, row in data[key].iterrows():
                x = np.array(row)
                a = ar(x)
                coeff1.append(a[1:][0].real)
                coeff2.append(a[1:][1].real)
                coeff3.append(a[1:][2].real)
                coeff4.append(a[1:][3].real)

            df_arburg[prefix + key + 'Coeff1'] = coeff1
            df_arburg[prefix + key + 'Coeff2'] = coeff2
            df_arburg[prefix + key + 'Coeff3'] = coeff3
            df_arburg[prefix + key + 'Coeff4'] = coeff4

    return df_arburg


def mean_freq(data, prefix, mag):
    """
    Weighted average of the frequency components to obtain a mean frequency
    """
    # built frequencies list (each column contain 128 value)
    # duration between each two successive captures is 0.02 s= 1/50hz

    df_max_indxs = pd.DataFrame()

    if mag:
        mean_freqs = []
        for i, row in data.iterrows():
            freqs = fftpack.fftfreq(len(row), d=0.02)
            # frequencies weighted sum
            mfreq = np.dot(freqs, row.values).sum() / float(row.values.sum())
            mean_freqs.append(mfreq)
        df_max_indxs[prefix] = mean_freqs
    else:
        for key in data.keys():
            mean_freqs = []
            for i, row in data[key].iterrows():
                freqs = fftpack.fftfreq(len(row), d=0.02)
                mfreq = np.dot(freqs, row.values).sum() / float(row.values.sum())
                mean_freqs.append(mfreq)
            df_max_indxs[prefix + key] = mean_freqs

    return df_max_indxs


def value_entropy(data, prefix):
    """
    value-entropy (entropy calculated from a histogram of
    quantization of the magnitude values to 20 bins)
    """
    entropies = []
    if data.isnull().values.any():
        # Handle NaN-Infite values of histogram
        for i, row in data.iterrows():
            entropies.append(0)
    else:
        for i, row in data.iterrows():
            count, hist = np.histogram(row, bins=20)
            entr = entropy(count)
            entropies.append(entr)
    df_value_entropy = pd.DataFrame({prefix: entropies})

    return df_value_entropy


def time_entropy(data, prefix):
    """
    time-entropy (entropy calculated from normalizing
    the magnitude signal and treating it as a probability
    distribution, which is designed to detect peakiness in
    time—sudden bursts of magnitude)
    """
    # TODO: This is not right implementation
    entropies = []
    for i, row in data.iterrows():
        # normalise data
        data_norm = pd.DataFrame(float(j) / sum(row.values) for j in row.values)
        entr = entropy(data_norm.value_counts())
        entropies.append(entr)
    df_time_entropy = pd.DataFrame({' '.join([prefix, 'Time Entropy']): entr}, index=[0])
    return df_time_entropy


def spectral_entropy(data, prefix):
    entropies = []
    for i, row in data.iterrows():
        psd = (np.abs(np.fft.fft(row)) ** 2)
        # Normalise the PSD as a probability density function
        data_norm = pd.DataFrame(float(i) / sum(psd) for i in psd)
        # calculate Shannon's entropy
        entr = entropy(data_norm.value_counts(), base=2)
        entropies.append(entr)
    df_spectral_entropy = pd.DataFrame({prefix: entropies})

    return df_spectral_entropy


def energy(data, prefix):
    def butter_bandpass(lowcut, highcut, fs):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(5, [low, high], btype='band')
        return b, a

    def butter_lowpass(cutoff, fs):
        nyq = 0.5 * fs
        cutoff = cutoff / nyq
        b, a = butter(5, cutoff, btype='low')
        return b, a

    def butter_highpass(cutoff, fs):
        nyq = 0.5 * fs
        cutoff = cutoff / nyq
        b, a = butter(5, cutoff, btype='high')
        return b, a

    def butter_filter(data, lowcut, highcut, fs, ftype):
        if ftype == 'low':
            b, a = butter_lowpass(lowcut, fs)
        elif ftype == 'high':
            b, a = butter_highpass(highcut, fs)
        else:
            b, a = butter_bandpass(lowcut, highcut, fs)

        y = lfilter(b, a, data)
        return y

    band0, band1, band2, band3, band4 = [], [], [], [], []
    for i, row in data.iterrows():
        s0 = butter_filter(row.values, lowcut=0.5, highcut=0, fs=30.0, ftype='low')
        s1 = butter_filter(row.values, lowcut=0.5, highcut=1.0, fs=30.0, ftype='band')
        s2 = butter_filter(row.values, lowcut=1.0, highcut=5.0, fs=30.0, ftype='band')
        s3 = butter_filter(row.values, lowcut=3.0, highcut=5.0, fs=30.0, ftype='band')
        s4 = butter_filter(row.values, lowcut=0, highcut=5.0, fs=30.0, ftype='high')

        # plt.plot(row, label='signal')
        # plt.plot(s0, label='0-0.5Hz')
        # plt.plot(s1, label='0.5-1Hz')
        # plt.plot(s2, label='1-35Hz')
        # plt.plot(s3, label='3-55Hz')
        # plt.plot(s4, label='>5Hz')
        # plt.legend(loc='upper left')
        # plt.show()
        # sys.exit()

        # Calculate power over each frequency band
        e0 = np.log(np.sum(np.abs(s0) ** 2))
        e1 = np.log(np.sum(np.abs(s1) ** 2))
        e2 = np.log(np.sum(np.abs(s2) ** 2))
        e3 = np.log(np.sum(np.abs(s3) ** 2))
        e4 = np.log(np.sum(np.abs(s4) ** 2))

        band0.append(e0)
        band1.append(e1)
        band2.append(e2)
        band3.append(e3)
        band4.append(e4)

    nrg = pd.DataFrame({''.join([prefix, 'Band0']): band0,
                        ''.join([prefix, 'Band1']): band1,
                        ''.join([prefix, 'Band2']): band2,
                        ''.join([prefix, 'Band3']): band3,
                        ''.join([prefix, 'Band4']): band4})

    return nrg


def angles(data, prefix, sensors):
    angls = []

    def angle(v1, v2, sp):
        # the cosinus value of the angle between Vector1 and Vector2
        cos_angle = sp / float(v1 * v2)
        # just in case some values were added automatically
        if cos_angle > 1:
            cos_angle = 1
        elif cos_angle < -1:
            cos_angle = -1
        # the angle value in radian
        angle_value = float(acos(cos_angle))
        return angle_value

    # TODO: angles for body-linear acceleration
    triaxial = []
    vectors = []

    for s in sensors:
        vector = []
        for index, (x, y, z) in enumerate(
                zip(data[s + 'X'].values, data[s + 'Y'].values, data[s + 'Z'].values)):
            print(s, x.shape, y.shape, z.shape)
            vector.append([x, y, z])
            # vector = np.array([x, y, z]).reshape(-1)
        vector = np.array(vector)
        # vector = vector.reshape(-1)
        vectors.append(vector)

    # ANGLES BETWEEN SENSORS
    res = [(v1, v2) for idx, v1 in enumerate(vectors) for v2 in vectors[idx + 1:]]
    for vector1, vector2 in res:
        # scalar product of Vector1 and Vector2
        scalar_product = np.dot(vector1, vector2)
        vector1_mag = sqrt((vector1 ** 2).sum())
        vector2_mag = sqrt((vector2 ** 2).sum())
        angle1 = angle(vector1_mag, vector2_mag, scalar_product)
        print(angle1)
    sys.exit()
    #
    # # ANGLES BETWEEN xAcc - Acc
    # vector1 = np.array([xAcc, np.zeros(xAcc.shape), np.zeros(xAcc.shape)]).reshape(-1)
    # vector2 = np.array([xAcc, yAcc, zAcc]).reshape(-1)
    # # scalar product of Vector1 and Vector2
    # scalar_product = np.dot(vector1, vector2)
    # vector1_mag = sqrt((vector1 ** 2).sum())
    # vector2_mag = sqrt((vector2 ** 2).sum())
    # angle2 = angle(vector1_mag, vector2_mag, scalar_product)
    #
    # # ANGLES BETWEEN yAcc - Acc
    # vector1 = np.array([np.zeros(xAcc.shape), yAcc, np.zeros(xAcc.shape)]).reshape(-1)
    # vector2 = np.array([xAcc, yAcc, zAcc]).reshape(-1)
    # # scalar product of Vector1 and Vector2
    # scalar_product = np.dot(vector1, vector2)
    # vector1_mag = sqrt((vector1 ** 2).sum())
    # vector2_mag = sqrt((vector2 ** 2).sum())
    # angle3 = angle(vector1_mag, vector2_mag, scalar_product)
    #
    # # ANGLES BETWEEN zAcc - Acc
    # vector1 = np.array([np.zeros(xAcc.shape), np.zeros(xAcc.shape), zAcc]).reshape(-1)
    # vector2 = np.array([xAcc, yAcc, zAcc]).reshape(-1)
    # # scalar product of Vector1 and Vector2
    # scalar_product = np.dot(vector1, vector2)
    # vector1_mag = sqrt((vector1 ** 2).sum())
    # vector2_mag = sqrt((vector2 ** 2).sum())
    # angle4 = angle(vector1_mag, vector2_mag, scalar_product)
    #
    # # ANGLES BETWEEN xGyro - Gyro
    # vector1 = np.array([xGyro, np.zeros(xGyro.shape), np.zeros(xGyro.shape)]).reshape(-1)
    # vector2 = np.array([xGyro, yGyro, zGyro]).reshape(-1)
    # # scalar product of Vector1 and Vector2
    # scalar_product = np.dot(vector1, vector2)
    # vector1_mag = sqrt((vector1 ** 2).sum())
    # vector2_mag = sqrt((vector2 ** 2).sum())
    # angle5 = angle(vector1_mag, vector2_mag, scalar_product)
    #
    # # ANGLES BETWEEN yGyro - Gyro
    # vector1 = np.array([np.zeros(xGyro.shape), yGyro, np.zeros(xGyro.shape)]).reshape(-1)
    # vector2 = np.array([xGyro, yGyro, zGyro]).reshape(-1)
    # # scalar product of Vector1 and Vector2
    # scalar_product = np.dot(vector1, vector2)
    # vector1_mag = sqrt((vector1 ** 2).sum())
    # vector2_mag = sqrt((vector2 ** 2).sum())
    # angle6 = angle(vector1_mag, vector2_mag, scalar_product)
    #
    # # ANGLES BETWEEN zGyro - Gyro
    # vector1 = np.array([np.zeros(xGyro.shape), np.zeros(xGyro.shape), zGyro]).reshape(-1)
    # vector2 = np.array([xGyro, yGyro, zGyro]).reshape(-1)
    # # scalar product of Vector1 and Vector2
    # scalar_product = np.dot(vector1, vector2)
    # vector1_mag = sqrt((vector1 ** 2).sum())
    # vector2_mag = sqrt((vector2 ** 2).sum())
    # angle7 = angle(vector1_mag, vector2_mag, scalar_product)
    #
    # angls.append([angle1, angle2, angle3, angle4, angle5, angle6, angle7])

    df_angles = pd.DataFrame(angls, columns=[prefix + '0', prefix + '1', prefix + '2', prefix + '3',
                                             prefix + '4', prefix + '5', prefix + '6'])
    return df_angles


def magnitude(data, sensors):
    df_magnitude = pd.DataFrame()
    # Sensor Magnitude
    for s in sensors:
        mag_values = []
        for index, (x, y, z) in enumerate(zip(data[s + 'X'].values, data[s + 'Y'].values, data[s + 'Z'].values)):
            # X, Y, Z Measurements in the Window
            win_mag = []
            for _, (elx, ely, elz) in enumerate(zip(x, y, z)):
                win_mag.append(np.sqrt(pow(elx, 2) + pow(ely, 2) + pow(elz, 2)))
            mag_values.append(win_mag)

        # Calculate Sensor Magnitude Features
        df_mag = pd.DataFrame(mag_values)
        df_mag_mean = means(df_mag, 'MeanAccMag', mag=True)
        df_mag_std = stds(df_mag, 'StdAccMag', mag=True)
        df_mag_mad = mad(df_mag, 'MadAccMag', mag=True)
        df_mag_percentile = percentile(df_mag, 'Percentile_AccMag', mag=True)
        df_mag_moment = moments(df_mag, 'Moment_AccMag', mag=True)
        df_mag_autocorr = autocorrelation(df_mag, 'AutoCorrAccMag')
        df_mag_min_max = min_max(df_mag, prefix=['MinAccMag', 'MaxAccMag'], mag=True)
        df_mag_kurtosis = kurtosis(df_mag, prefix='KurtosisAccMag', mag=True)
        df_mag_skewness = skewness(df_mag, prefix='SkewnessAccMag', mag=True)
        df_mag_iqr = interquartile(df_mag, prefix='IqrAccMag', mag=True)
        df_mag_value_entropy = value_entropy(df_mag, prefix="ValueEntropyMagAcc")
        # df_acc_mag_time_entropy = time_entropy(df_acc_mag, prefix="MagAccTimeEntropy")
        df_mag_spectral_entropy = spectral_entropy(df_mag, prefix="SpectralEntropyMagAcc")
        df_mag_energy = energy(df_mag, prefix="LogEnergyMagAcc")
        df_mag_arburg = arburg2(df_mag, order=4, prefix="ARMagAcc", mag=True)
        df_mean_freq = mean_freq(df_mag, prefix='MeanFreqMagAcc', mag=True)

        # Sensor Magnitude DataFrame
        df_sensor_magnitude = pd.concat([df_mag_mean, df_mag_std, df_mag_mad,
                                         df_mag_percentile, df_mag_moment, df_mag_min_max,
                                         df_mag_autocorr, df_mag_kurtosis, df_mag_skewness, df_mag_iqr,
                                         df_mag_value_entropy, df_mag_spectral_entropy, df_mag_energy,
                                         df_mag_arburg, df_mean_freq], axis=1)

        df_magnitude = pd.concat([df_magnitude, df_sensor_magnitude], axis=1)

    return df_magnitude


def process_labels(labels):
    """
    :param labels: Ground truth labels in DataFrame form
    :return:  Ground truth labels involving locomotion and activity in DataFrame form

    Description: Add locomotion context to the activity labeling
    """
    re_labels = []

    for l in labels['Label']:
        if '_' not in l:
            if 'washing' in l:
                l += '_standing'
            else:
                l += '_sitting'

        re_labels.append(l)

    re_labels = pd.DataFrame(re_labels, columns=['Label'])
    re_labels = re_labels['Label'].str.split(pat='_', expand=True)
    re_labels.columns = ['Activity', 'Locomotion']

    return re_labels


def features_extraction(user_data, config_sensors, path_to_save_features):
    """
    Args:
        user_data: str path to user data
        config_sensors: list of sensors
        path_to_save_features: str path to save extracted features

    Returns:
    """
    raw_signal = {}
    sensors = []
    for s in config_sensors:
        # Rename sensors
        sensors.append(s[0].upper() + s[1:])
        # read raw signal and store them in dictionary with key SensorAxes : eg. AccY
        raw_signal[s[0].upper() + s[1:] + 'X'] = pd.read_csv(os.path.join(user_data, s + '_x.csv'))
        raw_signal[s[0].upper() + s[1:] + 'Y'] = pd.read_csv(os.path.join(user_data, s + '_y.csv'))
        raw_signal[s[0].upper() + s[1:] + 'Z'] = pd.read_csv(os.path.join(user_data, s + '_z.csv'))
    labels = pd.read_csv(user_data + '/labels.csv')
    labels.columns = ['Label']
    # Reformat labels
    # ## Split Label into ADL and Locomotion
    labels = process_labels(labels)

    # ## Calculate features for each axes
    df_mean = means(raw_signal, prefix='Mean', mag=False)
    df_std = stds(raw_signal, prefix='Std', mag=False)
    df_mad = mad(raw_signal, prefix='Mad', mag=False)
    df_percentile = percentile(raw_signal, prefix='Percentile', mag=False)
    df_moment = moments(raw_signal, prefix='Moment', mag=False)
    df_min_max = min_max(raw_signal, prefix=['Min', 'Max'], mag=False)
    df_corr = correlation(raw_signal, prefix='Corr', sensors=sensors)
    df_cos = cosine_distance(raw_signal, prefix='CosDistance', sensors=sensors)
    df_kurtosis = kurtosis(raw_signal, prefix='Kurtosis', mag=False)
    df_skewness = skewness(raw_signal, prefix='Skewness', mag=False)
    df_iqr = interquartile(raw_signal, prefix='Iqr', mag=False)
    df_entropy = signal_entropy(raw_signal, prefix='Entropy')
    df_energy = energy_axial(raw_signal, prefix='Energy')
    df_sma = sma(raw_signal, prefix='Sma')
    df_arburg = arburg2(raw_signal, 4, 'AR', mag=False)
    df_mean_freq = mean_freq(raw_signal, prefix='MaxFreqInds', mag=False)
    # df_angles = angles(raw_signal, prefix='Angles', sensors=sensors)

    #  ## Calculate features for magnitude
    df_magnitude = magnitude(raw_signal, sensors=sensors)

    # Merge ALL the features
    features = pd.concat([df_mean, df_std, df_mad, df_min_max, df_corr, df_cos,
                          df_kurtosis, df_skewness, df_iqr, df_percentile, df_moment,
                          df_entropy, df_energy, df_sma, df_arburg, df_mean_freq,
                          df_magnitude, labels], axis=1)

    print(features.shape)
    features.to_csv(os.path.join(path_to_save_features, user_data.split('/')[-1] + '_features.csv'), index=False)


def load_data(path_to_data, sensors):
    """
    Args:
        path_to_data: str path to load pre-processed data
        sensors: list of sensors

    Returns: list of users
    """
    users = []
    for user in os.listdir(path_to_data):
        users.append(os.path.join(path_to_data, user))

    return users


def main():
    print("###########################")
    print("FEATURE EXTRACTION")
    print("###########################")
    print()

    # ## Parse configuration
    config = json.load(open('config.json'))
    path_to_data = os.path.join(config["path_to_data"], 'data')
    sensors = config["sensors"]

    #  ## Load pre-processed data
    users = load_data(path_to_data, sensors=sensors)

    path_to_save_features = os.path.join(os.path.join(config["path_to_data"]), 'features')
    make_sure_path_exists(path_to_save_features)

    # ## FEATURE EXTRACTION PROCESS
    for user in users:
        print('Process: ', user.split('/')[-1])
        features_extraction(user, config_sensors=sensors, path_to_save_features=path_to_save_features)


if __name__ == '__main__':
    main()
