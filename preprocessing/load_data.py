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
        raw = {s: pd.read_csv(os.path.join(path_to_data, activity, s + '.csv')) for s in sensors}
        rawData[activity] = raw

    return rawData


def parse_axes_data(path_to_data, sensors):
    """
    Args:
        path_to_data: str path to load single-axes sensor measurements
        sensors: list of str sensors to load raw measurements

    Returns: dictionary of DataFrames with single-axes sensor measurements

    """
    axesData = {}
    for activity in os.listdir(path_to_data):
        activityData = {}
        for s in sensors:
            activityData[s + '_x'] = pd.read_csv(os.path.join(path_to_data, activity, s+'_x.csv'))
            activityData[s + '_y'] = pd.read_csv(os.path.join(path_to_data, activity, s+'_y.csv'))
            activityData[s + '_z'] = pd.read_csv(os.path.join(path_to_data, activity, s+'_z.csv'))
        activityData['time'] = pd.read_csv(os.path.join(path_to_data, activity, 'time.csv'))
        axesData[activity] = activityData

    return axesData


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
