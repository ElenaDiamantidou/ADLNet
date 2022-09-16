"""
Project Name : active_data
Created on   : 02 Jul 2021
author       : ediamantidou
email        : ediamantidou@iti.gr
"""
import os, sys, errno

import datetime
import numpy as np
import pandas as pd
from scipy import signal


def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def median_filter(accData, gyroData, path, f_size=3):
    """
    :param accData: sync accelerometer signal
    :param gyroData: sync gyroscope signal
    :param path: path of the data
    :return: dataframe with sync filter signal

    Description: Apply median filter to the signal for noise reduction
    """

    def median(data):
        lgth, num_signal = data.shape
        f_data = np.zeros([lgth, num_signal])
        for i in range(num_signal):
            f_data[:, i] = signal.medfilt(data[:, i], f_size)
        return f_data

    dict_keys = list(accData.keys())
    for key in dict_keys:
        current_path = os.path.join('../', 'data', 'watch', 'medianData', path[23:], key)
        make_sure_path_exists(current_path)

        acc_data = accData[key]
        gyro_data = gyroData[key]
        acc_data_median = median(acc_data[['x', 'y', 'z']].values)
        gyro_data_median = median(gyro_data[['x', 'y', 'z']].values)

        acc_data_median = pd.DataFrame(acc_data_median, columns=['x', 'y', 'z'])
        acc_data_median['time'] = acc_data['time']
        gyro_data_median = pd.DataFrame(gyro_data_median, columns=['x', 'y', 'z'])
        gyro_data_median['time'] = gyro_data['time']
        path_save_vis = os.path.join(path[14:], key)

        # visualise_data.plot_raw_data(acc_data_median, gyro_data_median, 'median_data_visualisations', path_save_vis)
        # ## Save median filtered data
        acc_data_median.to_csv(current_path + '/accData.csv', index=False)
        gyro_data_median.to_csv(current_path + '/gyroData.csv', index=False)


def butterworth_filter(accData, gyroData, path):
    """
    :param accData: median filtered accelerometer signal
    :param gyroData: median filtered gyroscope signal
    :param path: path of teh data
    :return: dataframe with median + butterworth filter signal

    Description: Apply 3rd order low-pass filter with 20Hz cutOff frequency
    """

    def butterworth(data):
        """
        :param data: input signal after median filter
        :return: signal applied with 3rd order low-pass filter, 20Hz cutoff freq
        """
        # Creation of the filter
        cutOff = 20  # Cutoff freq
        sf = 50
        nyq = 0.5 * sf
        N = 3  # Filter order
        fc = cutOff / nyq  # Cutoff frequency normal
        b, a = signal.butter(N, fc, 'low')
        output = signal.filtfilt(b, a, data)

        return output

    dict_keys = list(accData.keys())
    for key in dict_keys:

        current_path = os.path.join('../', 'data', 'watch', 'butterworthData', path[25:], key)
        make_sure_path_exists(current_path)
        acc_data = accData[key]
        gyro_data = gyroData[key]

        acc_data_butter_x = butterworth(acc_data['x'])
        acc_data_butter_y = butterworth(acc_data['y'])
        acc_data_butter_z = butterworth(acc_data['z'])
        gyro_data_butter_x = butterworth(gyro_data['x'])
        gyro_data_butter_y = butterworth(gyro_data['y'])
        gyro_data_butter_z = butterworth(gyro_data['z'])

        acc_data_butter = pd.DataFrame({'x': acc_data_butter_x, 'y': acc_data_butter_y,
                                        'z': acc_data_butter_z, 'time': acc_data['time']})
        gyro_data_butter = pd.DataFrame({'x': gyro_data_butter_x, 'y': gyro_data_butter_y,
                                         'z': gyro_data_butter_z, 'time': gyro_data['time']})
        path_save_vis = os.path.join(path[16:], key)

        # visualise_data.plot_raw_data(acc_data_butter, gyro_data_butter, 'butterworth_data_visualisations', path_save_vis)
        # ## Save median+butterworth filtered data
        acc_data_butter.to_csv(current_path + '/accData.csv', index=False)
        gyro_data_butter.to_csv(current_path + '/gyroData.csv', index=False)

        # print(len(acc_data), len(gyro_data))


def segment_data(accData, gyroData, path, second=1):
    """
    :param accData: filtered accelerometer signal
    :param gyroData: filtered gyroscope signal
    :param path: path of teh data
    :param second: segmentation time
    :return: dataframe with sensor data of activity

    Description: Segmentation of data at 1 second window with 50% overlap
    """
    def segmentation(data):
        overlap = int(50*second*0.5)  # 50% overlap

        segmented = pd.DataFrame([data[:int(50*second)].values])
        duration = int(50*second)
        while duration < len(data):
            segment = pd.DataFrame([data[duration-overlap: duration+overlap].values])
            segmented = pd.concat([segmented, segment], axis=0)
            duration += overlap
        segmented = segmented.reset_index(drop=True)

        return segmented

    dict_keys = list(accData.keys())
    for key in dict_keys:
        current_path = os.path.join('../', 'data', 'watch', 'segmentData', str(second) + 's', path[30:], key)
        make_sure_path_exists(current_path)
        acc_x = segmentation(accData[key]['x'])
        acc_y = segmentation(accData[key]['y'])
        acc_z = segmentation(accData[key]['z'])
        gyro_x = segmentation(gyroData[key]['x'])
        gyro_y = segmentation(gyroData[key]['y'])
        gyro_z = segmentation(gyroData[key]['z'])

        time = pd.concat([accData[key]['time'], gyroData[key]['time']], axis=1)
        time.columns = ['acc_time', 'gyro_time']

        acc_x.to_csv(current_path + '/accData_x.csv', index=False)
        acc_y.to_csv(current_path + '/accData_y.csv', index=False)
        acc_z.to_csv(current_path + '/accData_z.csv', index=False)
        gyro_x.to_csv(current_path + '/gyroData_x.csv', index=False)
        gyro_y.to_csv(current_path + '/gyroData_y.csv', index=False)
        gyro_z.to_csv(current_path + '/gyroData_z.csv', index=False)
        time.to_csv(current_path + '/time.csv', index=False)


def concat_data(accData, gyroData, path):
    """
    :param accData: filtered accelerometer signal
    :param gyroData: filtered gyroscope signal
    :param path: path of teh data
    :return: dataframe with sensor data of activity

    Description: Concatenate data from one activity
    """

    dict_keys = list(accData.keys())
    acc_x, acc_y, acc_z = (pd.DataFrame() for _ in range(3))
    gyro_x, gyro_y, gyro_z = (pd.DataFrame() for _ in range(3))
    labels = []

    ext = os.path.join(path.split('/')[-3], path.split('/')[-2], path.split('/')[-1])
    current_path = os.path.join('../', 'data', 'watch', 'mergeData',  ext)
    make_sure_path_exists(current_path)

    for key in dict_keys:
        acc_data = accData[key]
        gyro_data = gyroData[key]
        len_of_samples = len(acc_data['x'])
        acc_x = pd.concat([acc_x, acc_data['x']], axis=0)
        acc_y = pd.concat([acc_y, acc_data['y']], axis=0)
        acc_z = pd.concat([acc_z, acc_data['z']], axis=0)
        gyro_x = pd.concat([gyro_x, gyro_data['x']], axis=0)
        gyro_y = pd.concat([gyro_y, gyro_data['y']], axis=0)
        gyro_z = pd.concat([gyro_z, gyro_data['z']], axis=0)

        # keep the label based on the ground truth label
        body_state = ['sitting', 'standing', 'walking']
        label = key.split('_')[2:]
        if True in [bs in label for bs in body_state]:
            if label[0] in body_state:
                # re-order to set body_state at the end
                # label[1:].append()
                activity = ' '.join(label[1:])
                locomotion = label[0]
                label = '_'.join([activity, locomotion])
        if len(label) == 1:
            label = label[0]
        if len(label) == 2 and label != 'answer_phone':
            label = '_'.join(label)
        labels.extend([label]*len_of_samples)
    labels = pd.DataFrame(labels)
    # print(labels.head())
    # print(acc_x.shape)

    # ## Save final form of the data data
    labels.to_csv(current_path + '/labels.csv', index=False, mode='a')
    acc_x.to_csv(current_path + '/acc_x.csv', index=False, mode='a')
    acc_y.to_csv(current_path + '/acc_y.csv', index=False, mode='a')
    acc_z.to_csv(current_path + '/acc_z.csv', index=False, mode='a')
    gyro_x.to_csv(current_path + '/gyro_x.csv', index=False, mode='a')
    gyro_y.to_csv(current_path + '/gyro_y.csv', index=False, mode='a')
    gyro_z.to_csv(current_path + '/gyro_z.csv', index=False, mode='a')


def save_data(activities_of_user, second):
    acc_x, acc_y, acc_z = (pd.DataFrame() for _ in range(3))
    gyro_x, gyro_y, gyro_z = (pd.DataFrame() for _ in range(3))
    labels = pd.DataFrame()

    user = activities_of_user[0].split('/')[-2]
    current_path = os.path.join('../', 'data', 'watch', 'data', str(second) + 's', user)
    print(current_path)
    make_sure_path_exists(current_path)

    for activity_data in activities_of_user:
        activity = activity_data.split('/')[-1]
        # Read the data and merge them
        acc_x = pd.concat([acc_x, pd.read_csv(activity_data + '/acc_x.csv')], axis=0)
        acc_y = pd.concat([acc_y, pd.read_csv(activity_data + '/acc_y.csv')], axis=0)
        acc_z = pd.concat([acc_z, pd.read_csv(activity_data + '/acc_z.csv')], axis=0)
        gyro_x = pd.concat([gyro_x, pd.read_csv(activity_data + '/gyro_x.csv')], axis=0)
        gyro_y = pd.concat([gyro_y, pd.read_csv(activity_data + '/gyro_y.csv')], axis=0)
        gyro_z = pd.concat([gyro_z, pd.read_csv(activity_data + '/gyro_z.csv')], axis=0)
        labels = pd.concat([labels, pd.read_csv(activity_data + '/labels.csv')], axis=0)

    # ## Save final form of the data
    acc_x.to_csv(current_path + '/acc_x.csv', index=False, mode='a')
    acc_y.to_csv(current_path + '/acc_y.csv', index=False, mode='a')
    acc_z.to_csv(current_path + '/acc_z.csv', index=False, mode='a')
    gyro_x.to_csv(current_path + '/gyro_x.csv', index=False, mode='a')
    gyro_y.to_csv(current_path + '/gyro_y.csv', index=False, mode='a')
    gyro_z.to_csv(current_path + '/gyro_z.csv', index=False, mode='a')
    labels.to_csv(current_path + '/labels.csv', index=False, mode='a')


def nearest_time(tms, df):
    # print(tms, df[np.abs(df - tms).argmin()])
    the_value = df[np.abs(df - tms).argmin()]
    # print(df[np.abs(df - tms).argmin()])
    the_index = df[df == the_value].index.values[0]

    return the_index


def get_sec_data(acc_data, gyro_data):
    """
    :param acc_data: DataFrame with acc data
    :param gyro_data: DataFrame with gyro data
    :return: dictionaries with one_sec_data

    Description: Group data by 1s to filter values (50Hz)
    """

    def add_value(data, tms):
        """
        :param data: DataFrame
        :param tms: timestamp
        :return: Interpolated DataFrame

        Description: Given a timestamp at the end of DataFrame, find an empty time to interpolate missing recordings
        """
        offset = datetime.timedelta(microseconds=1000)
        time_to_add = tms + offset
        while time_to_add in data['time']:
            time_to_add = tms - (offset + offset)
            offset += offset
        add = pd.DataFrame({'x': None, 'y': None, 'z': None, 'time': time_to_add}, index=[0])
        # Interpolate missing data
        data = pd.concat([data, add]).sort_values('time').reset_index(drop=True)
        # try:
        data = data.interpolate(method='pad')
        # except:
        #     data = data.interpolate(method='pad')

        return data

    acc_data = acc_data.set_index('time')
    gyro_data = gyro_data.set_index('time')

    acc_dict, gyro_dict = {}, {}
    # ## Get data for each second of recording
    for key, group in acc_data.groupby(pd.Grouper(freq='1s', origin=acc_data.index[0])):
        if len(group) > 45:
            acc_dict[key] = group.reset_index()

    for key, group in gyro_data.groupby(pd.Grouper(freq='1s', origin=acc_data.index[0])):
        if len(group) > 45:
            gyro_dict[key] = group.reset_index()

    # get the minimum length of data between acc and gyro
    if len(gyro_dict) < len(acc_dict):
        interpolation_range = len(gyro_dict)
    else:
        interpolation_range = len(acc_dict)

    # ## Interpolate data
    # 50 since 50Hz sampling rate
    for i in range(interpolation_range):
        if len(acc_dict[list(acc_dict.keys())[i]]) != 50 or len(gyro_dict[list(gyro_dict.keys())[i]]) != 50:
            if len(acc_dict[list(acc_dict.keys())[i]]) > 50:
                # remove data points from the activity beginning
                acc_diff = len(acc_dict[list(acc_dict.keys())[i]]) - 50

                acc_dict[list(acc_dict.keys())[i]] = acc_dict[list(acc_dict.keys())[i]].drop(
                    np.arange(acc_diff)).reset_index(drop=True)

            if len(gyro_dict[list(gyro_dict.keys())[i]]) > 50:
                # remove data points from the activity beginning
                gyro_diff = len(gyro_dict[list(gyro_dict.keys())[i]]) - 50
                gyro_dict[list(gyro_dict.keys())[i]] = gyro_dict[list(gyro_dict.keys())[i]].drop(
                    np.arange(gyro_diff)).reset_index(drop=True)

            while len(acc_dict[list(acc_dict.keys())[i]]) < 50:
                # add values at the and of the activity
                df = acc_dict[list(acc_dict.keys())[i]]
                acc_dict[list(acc_dict.keys())[i]] = add_value(df, df['time'].iloc[-2])

            while len(gyro_dict[list(gyro_dict.keys())[i]]) < 50:
                # add values at the and of the activity
                df = gyro_dict[list(gyro_dict.keys())[i]]
                gyro_dict[list(gyro_dict.keys())[i]] = add_value(df, df['time'].iloc[-2])

    # Merge Data to a Dataframe
    df_acc, df_gyro = pd.DataFrame(), pd.DataFrame()
    for key, sub_df in acc_dict.items():
        # Add your sub_df one by one
        df_acc = pd.concat([df_acc, sub_df], ignore_index=False)
    for key, sub_df in gyro_dict.items():
        # Add your sub_df one by one
        df_gyro = pd.concat([df_gyro, sub_df], ignore_index=False)

    df_acc = df_acc.reset_index(drop=True)
    df_gyro = df_gyro.reset_index(drop=True)

    # Right order of columns
    df_acc = df_acc[['x', 'y', 'z', 'time']]
    df_gyro = df_gyro[['x', 'y', 'z', 'time']]

    if len(df_acc['x']) != len(df_gyro['x']):
        diff = len(df_acc['x']) - len(df_gyro['x'])
        df_acc = df_acc[:-diff]

    return df_acc, df_gyro


def synchronise(accData, gyroData, path):
    """
    Args:
        accData: dictionary with current user acc data
        gyroData: dictionary with current user gyro data
        path: path to directory to save the sync data

    Returns: Sync and filtered data from both sensors

    """

    dict_keys = list(accData.keys())
    for key in dict_keys:
        acc_data = accData[key]
        gyro_data = gyroData[key]
        # ## Check if data are available and sensor fails were occurred
        if acc_data.shape[0] >= 100 and gyro_data.shape[0] >= 100:

            acc_data['msec'] = acc_data['time'].dt.microsecond//1000
            gyro_data['msec'] = gyro_data['time'].dt.microsecond//1000
            acc_data['sec'] = acc_data['time'].dt.second
            gyro_data['sec'] = gyro_data['time'].dt.second

            # print(sorted(dict(acc_data['second'].value_counts()).items()))
            # print(sorted(dict(gyro_data['second'].value_counts()).items()))

            # Drop dublicates that caused during the data collection
            acc_data = acc_data.drop_duplicates(subset='time', keep="first").reset_index(drop=True)
            gyro_data = gyro_data.drop_duplicates(subset='time', keep="first").reset_index(drop=True)

            # decide to drop the 1st
            if acc_data['msec'][1] - acc_data['msec'][0] > 10:
                acc_data = acc_data.drop([0]).reset_index(drop=True)

            sync_acc, sync_gyro = get_sec_data(acc_data[['x', 'y', 'z', 'time']], gyro_data[['x', 'y', 'z', 'time']])

            if sync_acc.shape[0] == sync_gyro.shape[0]:

                # ## Save sync data
                key = list(key)
                key[13] = ":"
                key[16] = ":"
                key = ''.join(key)

                # current_path = os.path.join('..', 'data', 'watch', 'syncData', path[22:], key)
                # make_sure_path_exists(current_path)
                #
                # sync_acc.to_csv(current_path + '/accData.csv', index=False)
                # sync_gyro.to_csv(current_path + '/gyroData.csv', index=False)

