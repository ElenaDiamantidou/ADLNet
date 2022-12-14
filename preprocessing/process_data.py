"""
Project Name : ADLNet
author       : Eleni Diamantides
email        : elenadiamantidou@gmail.com
"""

import os, sys, errno

import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal


def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def median_filter(rawData, user, activity, path, f_size=3):
    """
    Args:
       rawData: dictionary of DataFrames with current user (low pass filtered) raw data of dictionaries with multi-sensor measurements
       user: str containing the id of user
       activity: str of activity
       path: path to directory to save the sync data

    Returns: dictionary of DataFrames with median filtered signals

    Description: Apply median filter to the signal for noise reduction
                 Save filtered data into .csv files
                 Directory path_to_data/medianData
    """

    def median(data):
        lgth, num_signal = data.shape
        f_data = np.zeros([lgth, num_signal])
        for i in range(num_signal):
            f_data[:, i] = signal.medfilt(data[:, i], f_size)
        return f_data

    activity_keys = list(rawData.keys())
    for key in activity_keys:
        cur_path = os.path.join(path, 'medianData', user, activity, key)
        make_sure_path_exists(cur_path)
        sensor_keys = rawData[key].keys()
        activity_data = rawData[key]

        # Get raw measurements for each sensor
        for s in activity_data.keys():
            data_median = pd.DataFrame(activity_data[s], columns=['x', 'y', 'z'])
            data_median['time'] = activity_data[s]['time']
            data_median['unsync_time'] = activity_data[s]['unsync_time']

            filename = os.path.join(cur_path, s + '.csv')
            data_median.to_csv(filename, index=False)


def butterworth_filter(rawData, user, activity, path):
    """
    Args:
       rawData: dictionary of DataFrames with current user raw data of dictionaries with multi-sensor measurements
       user: str containing the id of user
       activity: str of activity
       path: path to directory to save the sync data

   Returns: dictionary of DataFrames with butterworth filtered signals

   Description: Apply 3rd order low-pass filter with 20Hz cutOff frequency
   """

    def butterworth(data):
        """
        Args:
            data: DataSeries of single axes raw measurements

        Returns: Numpy Array of low pass filtered signal

        Description: Signal applied with 3rd order low-pass filter, 20Hz cutoff freq
                     Save filtered data into .csv files
                     Directory path_to_data/butterworthData
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

    activity_keys = list(rawData.keys())
    for key in activity_keys:
        cur_path = os.path.join(path, 'butterworthData', user, activity, key)
        make_sure_path_exists(cur_path)
        activity_data = rawData[key]

        # Get raw measurements for each sensor
        for s in activity_data.keys():
            butter_x = butterworth(activity_data[s]['x'])
            butter_y = butterworth(activity_data[s]['y'])
            butter_z = butterworth(activity_data[s]['z'])

            data_butter = pd.DataFrame({'x': butter_x, 'y': butter_y, 'z': butter_z,
                                        'time': activity_data[s]['time'],
                                        'unsync_time': activity_data[s]['unsync_time']})

            filename = os.path.join(cur_path, s + '.csv')
            data_butter.to_csv(filename, index=False)


def segment_data(rawData, user, activity, path, second=1):
    """
    Args:
       rawData: dictionary of DataFrames with current user raw data of dictionaries with multi-sensor measurements
       user: str containing the id of user
       activity: str of activity
       path: path to directory to save the sync and filtered data
       second: int representing the segmentation window in seconds

    Returns: dictionary of DataFrames with butterworth filtered signals

    Description: Segmentation of data at specific second window with 50% overlap
                 Save segmented data into .csv files
                 Directory path_to_data/segmentData
    """

    def segmentation(data):
        """
        Args:
            data: DataSeries with single-axes sensor measurements

        Returns: DataFrame with segmented timeseries of single-axes sensor measurements
        """

        # ## Define 50% overlap window
        overlap = int(50 * second * 0.5)

        # ## Initialise a df with the segmented window
        segmented = pd.DataFrame([data[:int(50 * second)].values])
        # ## Defne the duration of the segment
        duration = int(50 * second)
        # ## Calculate segments with overlap
        while duration < len(data):
            segment = pd.DataFrame([data[duration - overlap: duration + overlap].values])
            segmented = pd.concat([segmented, segment], axis=0)
            duration += overlap
        segmented = segmented.reset_index(drop=True)

        return segmented

    activity_keys = list(rawData.keys())
    for key in activity_keys:
        cur_path = os.path.join(path, 'segmentData', str(second) + 's', user, activity, key)
        make_sure_path_exists(cur_path)
        activity_data = rawData[key]
        time_dict = {}

        # Get raw measurements for each sensor
        # Apply segmentation in ech axes separately
        for s in activity_data.keys():
            time_dict[s + '_unsync_time'] = activity_data[s]['unsync_time']
            time_dict['time'] = activity_data[s]['time']

            seg_x = segmentation(activity_data[s]['x'])
            filename = os.path.join(cur_path, s + '_x.csv')
            seg_x.to_csv(filename, index=False)

            seg_y = segmentation(activity_data[s]['y'])
            filename = os.path.join(cur_path, s + '_y.csv')
            seg_y.to_csv(filename, index=False)

            seg_z = segmentation(activity_data[s]['z'])
            filename = os.path.join(cur_path, s + '_z.csv')
            seg_z.to_csv(filename, index=False)

        time = pd.DataFrame(time_dict)
        filename = os.path.join(cur_path, 'time.csv')
        time.to_csv(filename, index=False)


def concat_data(axesData, user, activity, path, sensors):
    """
    Args:
       axesData: dictionary of DataFrames with current user raw data of dictionaries with single-axes sensor measurements
       user: str containing the id of user
       activity: str of activity
       path: path to directory to save the sync and filtered data merged
       sensors: list of sensors

    Returns: dictionary of DataFram    # cur_path = os.path.join(path, 'mergeData', user, activity)
es with butterworth filtered signals

    Description: Concatenate sensor data that represent the same activity
                 Save segmented data into .csv files
                 Directory path_to_data/mergeData
    """

    activity_keys = list(axesData.keys())
    print()
    print(user, activity)
    # initialise DataFrames to store data from the same activity
    data = {}
    for s in sensors:
        data[s] = {s + '_x': pd.DataFrame(), s + '_y': pd.DataFrame(), s + '_z': pd.DataFrame()}

    # x, y, z = {pd.DataFrame()}, {pd.DataFrame()}, {pd.DataFrame()}({s: pd.DataFrame()} for s in sensors)
    cur_path = os.path.join(path, 'mergeData', user, activity)
    make_sure_path_exists(cur_path)
    for key in activity_keys:
        activity_data = axesData[key]
        for s in sensors:
            data[s][s + '_x'] = pd.concat([data[s][s + '_x'], activity_data[s + '_x']], axis=0)
            data[s][s + '_y'] = pd.concat([data[s][s + '_y'], activity_data[s + '_y']], axis=0)
            data[s][s + '_z'] = pd.concat([data[s][s + '_z'], activity_data[s + '_z']], axis=0)

    labels = []
    for key in activity_keys:
        len_of_samples = len(axesData[key]['acc_x'])
        # define body states
        body_state = ['sitting', 'standing', 'walking', 'stand', 'sit', 'walk']
        # isolate label from the filename
        # keep the label based on the ground truth label
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
        else:
            label = label[0]

        labels.extend([label] * len_of_samples)

    # Create final DataFrame with the ground truth labels
    labels = pd.DataFrame(labels)
    labels.to_csv(cur_path + '/labels.csv', index=False)  # mode='a'
    # ## Save final form of the data
    for s in sensors:
        data[s][s + '_x'].to_csv(cur_path + '/' + s + '_x.csv', index=False)  # mode='a'
        data[s][s + '_y'].to_csv(cur_path + '/' + s + '_y.csv', index=False)  # mode='a'
        data[s][s + '_z'].to_csv(cur_path + '/' + s + '_z.csv', index=False)  # mode='a'


def save_data(activities_of_user, user, sensors, path):
    """
    Args:
        activities_of_user: list of directories with user data
        user: str containing the id of user
        sensors: list of sensors
        path: str path to directory to save final data

    Returns: None

    Description: Save final segmented and merged data into .csv files
                 Directory path_to_data/data

    """

    cur_path = os.path.join(path, 'data', user)
    make_sure_path_exists(cur_path)

    data = {}
    for s in sensors:
        data[s+'_x'] = pd.DataFrame()
        data[s+'_y'] = pd.DataFrame()
        data[s+'_z'] = pd.DataFrame()

    labels = pd.DataFrame()
    for activity_data in activities_of_user:
        labels = pd.concat([labels, pd.read_csv(os.path.join(activity_data, 'labels.csv'))])
        # print(labels)
        for s in sensors:
            data[s + '_x'] = pd.concat([data[s+'_x'], pd.read_csv(os.path.join(activity_data, s+'_x.csv'))], axis=0)
            data[s + '_y'] = pd.concat([data[s+'_y'], pd.read_csv(os.path.join(activity_data, s+'_y.csv'))], axis=0)
            data[s + '_z'] = pd.concat([data[s+'_z'], pd.read_csv(os.path.join(activity_data, s+'_z.csv'))], axis=0)

    # ## Save final form of the data
    labels.to_csv(os.path.join(cur_path, 'labels.csv'), index=False)
    for s in sensors:
        data[s + '_x'].to_csv(os.path.join(cur_path, s + '_x.csv'), index=False)
        data[s + '_y'].to_csv(os.path.join(cur_path, s + '_y.csv'), index=False)
        data[s + '_z'].to_csv(os.path.join(cur_path, s + '_z.csv'), index=False)


def nearest_time(tms, df):
    """
    Args:
        tms: integer timestamp
        df: DataFrame to search timestamp

    Returns: DataFrame index

    Description: search the nearest index timestamp to the input tms in the df
    """
    # print(tms, df[np.abs(df - tms).argmin()])
    the_value = df[np.abs(df - tms).argmin()]
    # print(df[np.abs(df - tms).argmin()])
    the_index = df[df == the_value].index.values[0]

    return the_index


def get_sync_data(sync_time, raw_data):
    """
    Args:
        sync_time: pandas data range containing time to sync
        raw_data: DataFrame with sensor unsync measurements

    Returns: dictionaries with one_sec_data

    Description: Sync raw sensor measurements
    """

    raw_data = raw_data.set_index('time')
    raw_data['sync'] = np.nan
    raw_data = raw_data.sort_index()
    sync_data = pd.DataFrame()
    x, y, z, time, unsync_time = [], [], [], [], []

    # ## The synchronisation is based on a common time range fixed in 50Hz
    for i, s in enumerate(sync_time):
        # Get the index with the nearest timestamp to the base timestamp
        nearest_tms_index = raw_data.index.get_indexer([s], method='nearest')[0]
        # just for analysis
        # raw_data['sync'].iloc[nearest_tms_index] = 'yes'

        x.append(raw_data.iloc[nearest_tms_index]['x'])
        y.append(raw_data.iloc[nearest_tms_index]['y'])
        z.append(raw_data.iloc[nearest_tms_index]['z'])
        time.append(s)
        unsync_time.append(raw_data.iloc[nearest_tms_index].name)

    sync_data = pd.DataFrame({'time': time, 'unsync_time': unsync_time,
                              'x': x, 'y': y, 'z': z})

    return sync_data


def synchronise(rawData, user, activity, path):
    """
    Args:
        rawData: dictionary with current user raw data of dictionaries with multi-sensor measurements
        user: str containing the id of user
        activity: str of activity
        path: path to directory to save the sync data

    Returns: Sync data from all raw sensor measurements

    """
    # TODO: optimise sync time

    activity_keys = list(rawData.keys())
    for key in activity_keys:
        cur_path = os.path.join(path, 'syncData', user, activity, key)
        make_sure_path_exists(cur_path)
        sensor_keys = rawData[key].keys()
        activity_data = rawData[key]

        # ## Check if data are available and sensor fails were not occurred
        # ## Process metrics with >= 100 sample recordings (100 samples equal to 2 sec of activity)
        if (len(activity_data[k]) >= 100 for k in sensor_keys):
            times = []
            for k in activity_data.keys():
                # ## Calculate time in msec & sec
                activity_data[k]['msec'] = activity_data[k]['time'].dt.microsecond // 1000
                activity_data[k]['sec'] = activity_data[k]['time'].dt.second

                # print(sorted(dict(activity_data[k]['sec'].value_counts()).items()))

                # Drop duplicates that caused during the data collection
                activity_data[k] = activity_data[k].drop_duplicates(subset='time', keep="first").reset_index(drop=True)
                times.append([activity_data[k]['time'].iloc[0], activity_data[k]['time'].iloc[-1]])

            # create time range with 50Hz freq
            sync_time = pd.date_range(start=min([i[0] for i in times]), end=max([i[1] for i in times]), freq='0.02S')

            # # decide to drop the 1st
            # if acc_data['msec'][1] - acc_data['msec'][0] > 10:
            #     acc_data = acc_data.drop([0]).reset_index(drop=True)

            for k in activity_data.keys():
                # print(sync_time)
                filename = os.path.join(cur_path, k + '.csv')
                syncData = get_sync_data(sync_time, activity_data[k][['x', 'y', 'z', 'time']])
                syncData.to_csv(filename, index=False)
