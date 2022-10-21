"""
Project Name : ADLNet
author       : Eleni Diamantides
email        : elenadiamantidou@gmail.com
"""
import sys, os
import json
import pandas as pd
import shutil

import load_data, process_data


# def statistic(acc, gyro):
#     # TODO: modify to work with users
#     print(len(acc[user]))
#     print(len(gyro[user]))
#     print(acc[user]['time'].max() - acc[user]['time'].min())
#     print(gyro[user]['time'].max() - gyro[user]['time'].min())
#     print()
#     visualise_data.plot_raw_data(acc[user], gyro[user], user)

def configuration():
    """
    Returns: Dictionary with configuration

    """
    # ## Parse configuration
    return json.load(open('../config.json'))


# string to boolean function
def str_to_bool(s):
    if s == "True":
        return True
    elif s == "False":
        return False
    else:
        return None


if __name__ == '__main__':

    config = configuration()
    path_to_data = config["path_to_data"]
    print('Process raw sensor data')
    path_to_raw_data = os.path.join(path_to_data, 'rawData')
    users = load_data.parde_dir(path_to_data=path_to_raw_data)
    usernames = [user.split('/')[-1] for user in users]

    # ## Rename the directories
    for user in usernames:
        # Parse paths for user activities
        activities_of_user = [os.path.join(path_to_raw_data, user, activity) for activity in
                              os.listdir(os.path.join(path_to_raw_data, user))]

        for activity in activities_of_user:
            # Load activity data
            for event in os.listdir(activity):
                # noob split
                date = event[:10]
                time = event[11:19]
                act = event[20:]

                # replace '-' with ':'
                time = time.replace("_", ":")

                re_event = "_".join([date, time, act])

                src = os.path.join(activity, event)
                dst = os.path.join(activity, re_event)

                # rename if necessary
                # shutil.move(src, dst)
                os.rename(src, dst)

    print("########################")
    print('Load and Synchronise Raw Data...')
    print()
    for user in usernames:
        # Parse paths for user activities
        activities_of_user = [os.path.join(path_to_raw_data, user, activity) for activity in
                              os.listdir(os.path.join(path_to_raw_data, user))]

        for activity in activities_of_user:
            # Load activity data
            # print(user, ":", activity.split('/')[-1])
            # Load raw sensor measurements
            rawData = load_data.main(activity, data_format=config["data_format"], sensors=config["sensors"])
            if str_to_bool(config["sync"]):
                # Synchronise data
                process_data.synchronise(rawData, user=user, activity=activity.split('/')[-1], path=path_to_data)

    #  ## Apply low-pass Butterworth filter
    print("########################")
    print('Apply Low-Pass Butterworth Filter...')
    print()
    if str_to_bool(config["butter"]):
        for user in usernames:
            path_to_sync_data = os.path.join(path_to_data, 'syncData')
            activities_of_user = [os.path.join(path_to_sync_data, user, activity) for activity in
                                  os.listdir(os.path.join(path_to_sync_data, user))]

            for activity in activities_of_user:
                rawData = load_data.main(activity, '.csv', sensors=config["sensors"])
                process_data.butterworth_filter(rawData, user=user, activity=activity.split('/')[-1], path=path_to_data)

    # ## Apply Median filter at sync data
    print("########################")
    print('Apply Median Filter...')
    print()
    if str_to_bool(config["median"]["flag"]):
        for user in usernames:
            path_to_sync_data = os.path.join(path_to_data, 'butterworthData')
            activities_of_user = [os.path.join(path_to_sync_data, user, activity) for activity in
                                  os.listdir(os.path.join(path_to_sync_data, user))]
            for activity in activities_of_user:
                rawData = load_data.main(activity, '.csv', sensors=config["sensors"])
                process_data.median_filter(rawData, user=user, activity=activity.split('/')[-1], path=path_to_data,
                                           f_size=config["median"]["size"])
    # # ## Segment data
    # print("########################")
    # print('Segmentation of data...')
    # print('Window: ', config["segmentation_window"], 's with 50% overlap')
    # print()
    # for user in usernames:
    #     print('Segmentation for:', user)
    #     path_to_filter_data = os.path.join(path_to_data, 'medianData')
    #     activities_of_user = [os.path.join(path_to_filter_data, user, activity) for activity in
    #                           os.listdir(os.path.join(path_to_filter_data, user))]
    #
    #     for activity in activities_of_user:
    #         rawData = load_data.main(activity, '.csv', sensors=config["sensors"])
    #         process_data.segment_data(rawData, user=user, activity=activity.split('/')[-1], path=path_to_data,
    #                                   second=config["segmentation_window"])

    # ## Concatenate data of the same activity
    # for user in usernames:
    #     path_to_segment_data = os.path.join(path_to_data, 'segmentData', str(config["segmentation_window"]) + 's')
    #
    #     activities_of_user = [os.path.join(path_to_segment_data, user, activity) for activity in
    #                           os.listdir(os.path.join(path_to_segment_data, user))]
    #     for activity in activities_of_user:
    #         axesData = load_data.parse_axes_data(activity, sensors=config["sensors"])
    #         process_data.concat_data(axesData, user=user, activity=activity.split('/')[-1], path=path_to_data,
    #                                  sensors=config["sensors"])

    # #  ## Save data at a final CSV
    # print()
    # print("########################")
    # print('Save data...')
    # print()
    # for user in usernames:
    #     path_to_axes_data = '../data/watch/mergeData/' + str(second) + 's/'
    #     activities_of_user = [os.path.join(path_to_axes_data, user, activity) for activity in
    #                           os.listdir(path_to_axes_data + user)]
    #     process_data.save_data(activities_of_user, second)

    print()
    print('Done.')
