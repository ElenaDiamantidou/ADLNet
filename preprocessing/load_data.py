"""
Project Name : ADLNet
author       : Eleni Diamantides
email        : elenadiamantidou@gmail.com
"""

import os, sys, errno
import pandas as pd


def parde_dir(path_to_data):
    """
    Args:
        path_to_data: string containing the path to load raw sensor measurements from multiple users

    Returns: list of string paths from multiple user raw sensor measurements

    """
    directories = []

    for directory in os.listdir(path_to_data):
        directories.append(os.path.join(path_to_data, directory))
    return directories


def parse_data(second):
    path_to_data = '../data/watch/data/' + str(second) + 's'
    directories = []

    for directory in os.listdir(path_to_data):
        directories.append(os.path.join(path_to_data, directory))

    return directories


def parse_raw_data(path_to_data, sensors):
    """
    Args:
        path_to_data: str path to load raw data
        sensors: list of str sensors to load raw measurements

    Returns: dictionary of activities of raw sensor measurements

    """
    rawData = {}
    activity_label = path_to_data.split('/')[-1]

    for activity in os.listdir(path_to_data):
        try:
            raw = {s: pd.read_csv(os.path.join(path_to_data, activity, s+'Data.txt'),
                                  delimiter=' ', header=None) for s in sensors}

            for s in raw.keys():
                raw[s].columns = ['x', 'y', 'z', 'time']
                raw[s]['time'] = pd.to_datetime(raw[s]['time'])
                raw[s]['x'] = raw[s]['x'].str.replace(',', '.').astype(float)
                raw[s]['y'] = raw[s]['y'].str.replace(',', '.').astype(float)
                raw[s]['z'] = raw[s]['z'].str.replace(',', '.').astype(float)
                # print(os.path.join(path_to_data, activity, s+'Data.txt'))
                # print(raw[s].head())
            rawData[activity] = raw
        except:
            pass

    return rawData



def parse_csv_raw_data(path_to_data, sensors):
    """
    Args:
        path_to_data: str path to load raw data
        sensors: list of str sensors to load raw measurements

    Returns: dictionary of activities of raw sensor measurements

    """
    rawData = {}
    activity_label = path_to_data.split('/')[-1]
    for activity in os.listdir(path_to_data):
        raw = {s: pd.read_csv(os.path.join(path_to_data, activity, s + 'Data.csv')) for s in sensors}
        for s in raw.keys():
            raw[s].columns = ['x', 'y', 'z', 'time']
            raw[s]['time'] = pd.to_datetime(raw[s]['time'])

        rawData[activity] = raw

    return rawData


def parse_gyro_data(path_to_data):
    gyroData = {}
    for activity in os.listdir(path_to_data):
        data = pd.read_csv(os.path.join(path_to_data, activity, 'gyrData.txt'), delimiter=' ', header=None)
        data.columns = ['x', 'y', 'z', 'time']
        data['x'] = data['x'].str.replace(',', '.').astype(float)
        data['y'] = data['y'].str.replace(',', '.').astype(float)
        data['z'] = data['z'].str.replace(',', '.').astype(float)

        data['time'] = pd.to_datetime(data['time'])
        gyroData[activity] = data

    return gyroData


def parse_axes_data(path_to_data):
    accData, gyroData = {}, {}
    for activity in os.listdir(path_to_data):
        acc_x = pd.read_csv(os.path.join(path_to_data, activity, 'accData_x.csv'))
        acc_y = pd.read_csv(os.path.join(path_to_data, activity, 'accData_y.csv'))
        acc_z = pd.read_csv(os.path.join(path_to_data, activity, 'accData_z.csv'))
        gyro_x = pd.read_csv(os.path.join(path_to_data, activity, 'gyroData_x.csv'))
        gyro_y = pd.read_csv(os.path.join(path_to_data, activity, 'gyroData_y.csv'))
        gyro_z = pd.read_csv(os.path.join(path_to_data, activity, 'gyroData_z.csv'))
        data = {'x': acc_x, 'y': acc_y, 'z': acc_z}
        accData[activity] = data
        data = {'x': gyro_x, 'y': gyro_y, 'z': gyro_z}
        gyroData[activity] = data

    return accData, gyroData


def main(activity_data, data_format, sensors):
    """
    Args:
        activity_data: str of path to load user activity raw measurement
        data_format: str of data format [.csv or .txt]
        sensors: list of str of raw sensor measurements to load

    Returns: dictionary of raw measurements from a user for multiple events of specific activity

    """
    if data_format == '.csv':
        data = parse_csv_raw_data(activity_data, sensors)
    else:
        data = parse_raw_data(activity_data, sensors)

    return data
