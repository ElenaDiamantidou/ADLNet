"""
Project Name : ADLNet
author       : Eleni Diamantides
email        : elenadiamantidou@gmail.com
"""
import sys, os
import json
import pandas as pd

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


if __name__ == '__main__':

    config = configuration()
    path_to_data = config["path_to_data"]
    print('Process raw sensor data')
    path_to_raw_data = os.path.join(path_to_data, 'rawData')
    users = load_data.parde_dir(path_to_data=path_to_raw_data)
    usernames = [user.split('/')[-1] for user in users]

    print("########################")
    print('Synchronise Accelerometer & Gyroscope Data...')
    print()
    for user in usernames:
        # Parse paths for user activities
        activities_of_user = [os.path.join(path_to_raw_data, user, activity) for activity in
                              os.listdir(os.path.join(path_to_raw_data, user))]

        for activity in activities_of_user:
            # Load activity data
            print(user, activity)
            rawData = load_data.main(activity, data_format=config["data_format"], sensors=config["sensors"])
            # Synchronise data
            process_data.synchronise(rawData, user=user, path=path_to_data)
            sys.exit()

    # # ## Apply Median filter at sync data
    # print("########################")
    # print('Apply Median Filter...')
    # print()
    # input()
    #
    # for user in usernames:
    #     path_to_sync_data = '../data/watch/syncData/'
    #     activities_of_user = [os.path.join(path_to_sync_data, user, activity) for activity in
    #                           os.listdir(path_to_sync_data + user)]
    #     for activity in activities_of_user:
    #         accData, gyroData = load_data.main(activity, '.csv')
    #         process_data.median_filter(accData, gyroData, path=activity)
    #
    # #  ## Apply low-pass Butterworth filter
    # print("########################")
    # print('Apply Low-Pass Butterworth Filter...')
    # print()
    # for user in usernames:
    #     path_to_sync_median_data = '../data/watch/medianData/'
    #     activities_of_user = [os.path.join(path_to_sync_median_data, user, activity) for activity in
    #                           os.listdir(path_to_sync_median_data + user)]
    #     for activity in activities_of_user:
    #         accData, gyroData = load_data.main(activity, '.csv')
    #         process_data.butterworth_filter(accData, gyroData, path=activity)
    #
    # # ## Segment data
    # print("########################")
    # print('Segmentation of data...')
    # print('Window: ', second, 's with 50% overlap')
    # print()
    # for user in usernames:
    #     print('Segmentation for:', user)
    #     path_to_sync_butter_data = '../data/watch/butterworthData/'
    #     activities_of_user = [os.path.join(path_to_sync_butter_data, user, activity) for activity in
    #                           os.listdir(path_to_sync_butter_data + user)]
    #     for activity in activities_of_user:
    #         accData, gyroData = load_data.main(activity, '.csv')
    #         process_data.segment_data(accData, gyroData, path=activity, second=second)
    #
    # ## Concatenate data of the same activity
    # for user in usernames:
    #     path_to_segment_data = '../data/watch/segmentData/' + str(second) + 's/'
    #     activities_of_user = [os.path.join(path_to_segment_data, user, activity) for activity in
    #                           os.listdir(path_to_segment_data + user)]
    #     for activity in activities_of_user:
    #         accData, gyroData = load_data.parse_axes_data(activity)
    #         process_data.concat_data(accData, gyroData, path=activity)
    #
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
