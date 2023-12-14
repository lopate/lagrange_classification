import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm

def load_indices():
    list_of_files = ['./code/PAMAP2_Dataset/Protocol/subject101.dat',
                     './code/PAMAP2_Dataset/Protocol/subject102.dat',
                     './code/PAMAP2_Dataset/Protocol/subject103.dat',
                     './code/PAMAP2_Dataset/Protocol/subject104.dat',
                     './code/PAMAP2_Dataset/Protocol/subject105.dat',
                     './code/PAMAP2_Dataset/Protocol/subject106.dat',
                     './code/PAMAP2_Dataset/Protocol/subject107.dat',
                     './code/PAMAP2_Dataset/Protocol/subject108.dat',
                     './code/PAMAP2_Dataset/Protocol/subject109.dat' ]

    subjectID = [1,2,3,4,5,6,7,8,9]

    activityIDdict = {0: 'transient',
                  1: 'lying',
                  2: 'sitting',
                  3: 'standing',
                  4: 'walking',
                  5: 'running',
                  6: 'cycling',
                  7: 'Nordic_walking',
                  9: 'watching_TV',
                  10: 'computer_work',
                  11: 'car driving',
                  12: 'ascending_stairs',
                  13: 'descending_stairs',
                  16: 'vacuum_cleaning',
                  17: 'ironing',
                  18: 'folding_laundry',
                  19: 'house_cleaning',
                  20: 'playing_soccer',
                  24: 'rope_jumping' }

    colNames = ["timestamp", "activityID","heartrate"]

    IMUhand = ['handTemperature', 
               'handAcc16_1', 'handAcc16_2', 'handAcc16_3', 
               'handAcc6_1', 'handAcc6_2', 'handAcc6_3', 
               'handGyro1', 'handGyro2', 'handGyro3', 
               'handMagne1', 'handMagne2', 'handMagne3',
               'handOrientation1', 'handOrientation2', 'handOrientation3', 'handOrientation4']

    IMUchest = ['chestTemperature', 
               'chestAcc16_1', 'chestAcc16_2', 'chestAcc16_3', 
               'chestAcc6_1', 'chestAcc6_2', 'chestAcc6_3', 
               'chestGyro1', 'chestGyro2', 'chestGyro3', 
               'chestMagne1', 'chestMagne2', 'chestMagne3',
               'chestOrientation1', 'chestOrientation2', 'chestOrientation3', 'chestOrientation4']

    IMUankle = ['ankleTemperature', 
               'ankleAcc16_1', 'ankleAcc16_2', 'ankleAcc16_3', 
               'ankleAcc6_1', 'ankleAcc6_2', 'ankleAcc6_3', 
               'ankleGyro1', 'ankleGyro2', 'ankleGyro3', 
               'ankleMagne1', 'ankleMagne2', 'ankleMagne3',
               'ankleOrientation1', 'ankleOrientation2', 'ankleOrientation3', 'ankleOrientation4']

    columns = colNames + IMUhand + IMUchest + IMUankle  #all columns in one list
    return list_of_files, subjectID, activityIDdict, colNames, IMUhand, IMUchest, IMUankle, columns

def formDataFrame(list_of_files, columns):
    dataCollection = pd.DataFrame()
    for file in list_of_files:
        procData = pd.read_table(file, header=None, sep='\s+')
        procData.columns = columns
        procData['subject_id'] = int(file[-5])
        dataCollection = pd.concat((dataCollection, procData), ignore_index=True)
    dataCollection.reset_index(drop=True, inplace=True)
    return dataCollection

def dataCleaning(dataCollection, selected_features, activityID, activityIDdict):
    dataCollection = dataCollection[selected_features]
    for i in list(activityIDdict.keys()):
        if i not in activityID:
            dataCollection = dataCollection.drop(dataCollection[dataCollection.activityID == i].index)
    dataCollection = dataCollection.interpolate()
    return dataCollection

def set_trajectory_target(subjectID, activityID, selected_sensors, dataCol):
    all_velocities, targets = [], []
    for user_id in subjectID:
        for activity_id in activityID:
            df = dataCol.loc[(dataCol.subject_id == user_id) & (dataCol.activityID == activity_id)]
            velocities = df[selected_sensors].to_numpy()
            l = len(velocities) // 500
            if l > 0:
                for sub_trajectory in np.array_split(velocities, l, axis = 0):
                    all_velocities.append(sub_trajectory)
                    targets.append(activity_id)
    return all_velocities, targets

tau = 0.01

def vel2coord(vel_components):
    
    coords = [[tau / 3 * (vel_arr[i - 1] + 4 * vel_arr[i] + vel_arr[i + 1]) for i in range(1, len(vel_arr) - 1)] \
                for vel_arr in vel_components.T]
    return np.array(coords).T

def vel2acc(vel_components):
    
    accels = [[(vel_arr[i + 1] - vel_arr[i - 1]) / (2 * tau) for i in range(1, len(vel_arr) - 1)] \
                for vel_arr in vel_components.T]
    return np.array(accels).T

def vel_cut(vel_components):
    return vel_components[1 : (-1)]


def form_trajectory(vel_components, name):
    X_train = np.hstack((vel2coord(vel_components), vel_cut(vel_components)))
    y_train = vel2acc(vel_components)
    with open(Path("code/trajectories", name), 'wb') as handle:
        pickle.dump([X_train, y_train], handle, protocol=pickle.HIGHEST_PROTOCOL)
    
def form_all_trajectories(multiple_vel_arrs, experiment_name):
    
    for i in tqdm(range(len(multiple_vel_arrs))):
        
        vel_components = multiple_vel_arrs[i]
        
        name = experiment_name + '_' + str(i + 1) + '.pickle'
        form_trajectory(vel_components, name)