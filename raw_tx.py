# -*- coding: utf-8 -*-
"""
Created on Tue Apr 03 17:22:06 2018

@author: research
"""

from __future__ import division
import argparse
import datetime
import numpy as np
import math
#import time

#PUT AN END FOR INFINITE LOOP THIS IS GOING INTO INFINITE LOOP MOSTLY!!
def peak(data, time):
    greatest = np.max(data)
    threshold = 0.9*greatest
    peaks = [i>threshold for i in data]
    no_of_peaks = sum(peaks)
    no_of_attempts = 70
    while no_of_peaks<3 and no_of_attempts:
        threshold -= 0.1
        peaks = [i>threshold for i in data]
        no_of_peaks = sum(peaks)
        no_of_attempts -= 1
        #print "no of peaks:", no_of_peaks, "for data", data
    if no_of_attempts==0:
        threshold = 0.2
        peaks = [i>threshold for i in data]
        no_of_peaks = sum(peaks)
    first = peaks.index(True)
    prev_time = time[first]
    prev_i = first
    timesec = 0
    #timesum = datetime.datetime(0,0,0,0,0)
    timediff = []
    #this_time = 0
    for i in range(first, len(data)):
        if i:
            timesum = time[i]-prev_time
            timesec += (i-prev_i)
            timediff.append(timesum.total_seconds())
            prev_time = time[i]
            prev_i = i
    #return sum(timediff)/no_of_peaks
    return timesec*10/no_of_peaks

def resultant(vecx, vecy, vecz):
    res_vec = []
    for i in range(len(vecx)):
        res = math.sqrt(vecx[i]**2 + vecy[i]**2 + vecz[i]**2)
        res_vec.append(res)
    return np.mean(res_vec)


def mad(data, axis=None): #mean absolute deviation
    return np.mean(np.absolute(data - np.mean(data, axis)), axis)


parser = argparse.ArgumentParser()
parser.add_argument('raw',type = str, help = 'raw file location')
#parser.add_argument('tx', type =str, help = 'transformed output file')
args = parser.parse_args()

raw_file = open(args.raw, 'rb')
tx_file = open(args.raw+'_transformed.txt', 'wb')

activity_list = ["Walking","Jogging","Stairs","Sitting","Standing","LyingDown"]
bins = [-2.5,0,2.5,5,7.5,10,12.5,15,17.5,20]

raw_window = []
acc = 0  #if accumulate is true, accumulate the statistical feats
prev_usr = -1
prev_time = -1
prev_act = ""
val_list_x = []
val_list_y = []
val_list_z = []
timestamps = []
for line in raw_file:
    if line:
        try:
            line = line.replace(';', '\n')
            line = line.rstrip()
            print args.raw
            try:
                split_line = line.split(',')
                if len(split_line) == 6:
                    usr, act, t, x, y, z = split_line
                elif len(split_line) > 6:
                    usr = split_line[0]
                    act = split_line[1]
                    t = split_line[2]
                    x = split_line[3]
                    y = split_line[4]
                    z = split_line[5]
                #print type(t), t
                if t=='0':
                    continue
                if z=='':
                    continue
                else:
                    try:
                        float(z) and float(y) and float(x)
                    except Exception as e:
                        print e
                        print split_line
                        continue
            except Exception as e:
                print line
                print e
                continue
            userid = int(usr)
            try:
                time = datetime.datetime.fromtimestamp(int(t)/1000)
            except Exception as e:
                print e
                continue
            if prev_time==-1:
                prev_time = time
            time_diff = time-prev_time
            #print time_diff, userid
            #timee = time.gmtime(int(t))
            #either file is starting to process or user is previous and no gap, if more than 15 seconds that means gap, then reset!
            #if acc<200 and (prev_usr==-1 or (prev_usr==userid and time_diff.total_seconds()<15 and act in activity_list and prev_act==act)):
            #if this isnt training set, then we just can club on the basis of time :)
            #if acc<200 and (prev_usr==-1 or (prev_usr==userid and time_diff.total_seconds()<15)):
            #for the AR dataset:
            if acc<200 and (prev_usr==-1 or (prev_usr==userid and prev_act==act)):
                if prev_usr==-1:
                    prev_usr = userid
                    prev_time = time
                    prev_act = act
                elif userid==prev_usr:
                    acc += 1
                    val_list_x.append(float(x))
                    val_list_y.append(float(y))
                    val_list_z.append(float(z))
                    timestamps.append(time)
            else:
                if acc==200:
                #set of transformations
                    attribute = [0.0]*45
                    bin_list_x = list(np.digitize(val_list_x, bins))
                    bin_list_y = list(np.digitize(val_list_y, bins))
                    bin_list_z = list(np.digitize(val_list_z, bins))
                    for i in range(10):
                        attribute[i] = bin_list_x.count(i)/200
                        attribute[10+i] = bin_list_y.count(i)/200
                        attribute[20+i] = bin_list_z.count(i)/200
                    sum_x = sum(val_list_x)
                    sum_y = sum(val_list_y)
                    sum_z = sum(val_list_z)
                    attribute[30] = np.mean(val_list_x)
                    attribute[31] = np.mean(val_list_y)
                    attribute[32] = np.mean(val_list_z)
                    attribute[33] = peak(val_list_x, timestamps)
                    attribute[34] = peak(val_list_y, timestamps)
                    attribute[35] = peak(val_list_z, timestamps)
                    attribute[36] = mad(val_list_x)
                    attribute[37] = mad(val_list_y)
                    attribute[38] = mad(val_list_z)
                    attribute[39] = np.std(val_list_x)
                    attribute[40] = np.std(val_list_y)
                    attribute[41] = np.std(val_list_z)
                    attribute[42] = resultant(val_list_x, val_list_y, val_list_z)
                    attribute[43] = prev_act
                    attribute[44] = timestamps[-1]
                    #print userid, attribute
                    #write to transformed file
                    tx_file.write(str(userid)+",")
                    for attr in attribute[:-1]:
                        tx_file.write(str(attr)+",")
                    tx_file.write(str(attribute[-1]))
                    tx_file.write("\n")
                #raw_input()
                #print "acc value:" ,acc
                acc = 0
                val_list_x[:] = []
                val_list_y[:] = []
                val_list_z[:] = []
                timestamps[:] = []
                prev_usr = userid
                prev_time = time
                prev_act = act
                val_list_x.append(float(x))
                val_list_y.append(float(y))
                val_list_z.append(float(z))
                timestamps.append(time)
        except Exception as e:
            print e
            #raw_input()
            continue

print args.raw, "processed!!"
