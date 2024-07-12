# Pickle function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from datetime import datetime
from tqdm import tqdm as tqdm


def PickleData(dataPath, trainPath, valPath, testPath, outTrainPath, outTestPath, outValPath):
    ## PARAMS
    LAT, LON, SOG, HEADING, TIMESTAMP, MMSI, PATH = list(range(7))
    #Data path goes here. These are the file you are giving the program to pickle.
    #If you use "" then the file will not be pickled. 
    #Example: PickleData(r"C:\Users\Rael\Desktop\AISweden\DataDir\", "data_train.csv", "", "",
    #                            "data_train.pkl", "", "")
    #The above example will only pickle on data_train.

    dataset_path = dataPath
    l_csv_filename =[trainPath,
                    valPath,
                    testPath]

    pkl_filename_train = outTrainPath
    pkl_filename_valid = outValPath
    pkl_filename_test  = outTestPath

    #========================================================================
    ## LOADING CSV FILES
    #========================================================================
    l_l_msg = [] # list of AIS messages, each row is a message (list of AIS attributes)
    m_msg_train = []
    m_msg_valid = []
    m_msg_test = []
    for csv_filename in l_csv_filename:
        if(csv_filename == ""):
            continue
        data_path = os.path.join(dataset_path,csv_filename)
        f = pd.read_csv(data_path)
        print("Reading ", csv_filename, "...")
        for i in range(len(f["mmsi"])):
            row = f.iloc[i]
            # Note: If your columns have other names, update them here
            l_l_msg.append([float(row["alat"]), float(row["alon"]), float(row["aspeed"]),
                                float(row["aheading"]),
                                int(row["timestamp"]), int(row["mmsi"]),
                            int(row["pathIni"])])
        if csv_filename == l_csv_filename[0]:
            m_msg_train = np.array(l_l_msg)
        if csv_filename == l_csv_filename[1]:
            m_msg_valid = np.array(l_l_msg)
        if csv_filename == l_csv_filename[2]:
            m_msg_test = np.array(l_l_msg)

    print("Number of msgs in the training set: ",len(m_msg_train))
    print("Number of msgs in the validation set: ",len(m_msg_valid))
    print("Number of msgs in the test set: ",len(m_msg_test))


    ## MERGING INTO DICT
    #======================================
    # Creating AIS tracks from the list of AIS messages.
    # Each AIS track is formatted by a dictionary.
    print("Convert to dicts of vessel's tracks...")


    # Training set
    if(len(m_msg_train) != 0):
        Vs_train = dict()
        for v_msg in tqdm(m_msg_train):
            mmsi = int(v_msg[MMSI])
            pathIni = int(v_msg[PATH])
            if not (pathIni in list(Vs_train.keys())):
                Vs_train[pathIni] = {'mmsi': mmsi, 'traj': np.empty((0,6))}
            Vs_train[pathIni] = {'mmsi': mmsi, 'traj': np.concatenate((Vs_train[pathIni]['traj'], np.expand_dims(v_msg[:6],0)), axis = 0)}
        for key in tqdm(list(Vs_train.keys())):
            mmsi = int(Vs_train[key]['traj'][0][5])
            Vs_train[key] = {'mmsi': mmsi,'traj': np.array(sorted(Vs_train[key]['traj'], key=lambda m_entry: m_entry[TIMESTAMP]))}

    # Validation set
    if(len(m_msg_valid) != 0):
        Vs_valid = dict()
        for v_msg in tqdm(m_msg_valid):
            mmsi = int(v_msg[MMSI])
            pathIni = int(v_msg[PATH])
            if not (pathIni in list(Vs_valid.keys())):
                Vs_valid[pathIni] = {'mmsi': mmsi, 'traj': np.empty((0,6))}
            Vs_valid[pathIni] = {'mmsi': mmsi, 'traj': np.concatenate((Vs_valid[pathIni]['traj'], np.expand_dims(v_msg[:6],0)), axis = 0)}
        for key in tqdm(list(Vs_valid.keys())):
            mmsi = int(Vs_valid[key]['traj'][0][5])
            Vs_valid[key] = {'mmsi': mmsi,'traj': np.array(sorted(Vs_valid[key]['traj'], key=lambda m_entry: m_entry[TIMESTAMP]))}

    # Test set
    if(len(m_msg_test) != 0):
        Vs_test = dict()
        for v_msg in tqdm(m_msg_test):
            mmsi = int(v_msg[MMSI])
            pathIni = int(v_msg[PATH])
            if not (pathIni in list(Vs_test.keys())):
                Vs_test[pathIni] = {'mmsi': mmsi, 'traj': np.empty((0,6))}
            Vs_test[pathIni] = {'mmsi': mmsi, 'traj': np.concatenate((Vs_test[pathIni]['traj'], np.expand_dims(v_msg[:6],0)), axis = 0)}
        for key in tqdm(list(Vs_test.keys())):
            mmsi = int(Vs_test[key]['traj'][0][5])
            Vs_test[key] = {'mmsi': mmsi,'traj': np.array(sorted(Vs_test[key]['traj'], key=lambda m_entry: m_entry[TIMESTAMP]))}


    ## PICKLING
    #======================================
    for filename, filedict in zip([pkl_filename_train,pkl_filename_valid,pkl_filename_test],
                                [Vs_train,Vs_valid,Vs_test]):
        if(filename != ""):
            print("Writing to ", os.path.join(dataset_path,filename),"...")
            with open(filename,"wb") as f:
                pickle.dump(filedict,f)
            print("Total number of tracks: ", len(filedict))
