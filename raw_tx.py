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

parser = argparse.ArgumentParser()
parser.add_argument('raw',type = str, help = 'raw file location')
parser.add_argument('tx', type =str, help = 'transformed output file')
args = parser.parse_args()

raw_file = open(args.raw, 'rb')
tx_file = open(args.tx, 'wb')

activity_list = ["Walking","Jogging","Stairs","Sitting","Standing","LyingDown"]
bins = [-2.5,0,2.5,5,7.5,10,12.5,15,17.5,20]

raw_window = []
acc = 0  #if accumulate is true, accumulate the statistical feats
prev_usr = -1
prev_time = -1
val_list_x = []
val_list_y = []
val_list_z = []
for line in raw_file:
    line = line.replace(';', ',')
    #print line
    usr, act, t, x, y, z, __ = line.split(',')    
    userid = int(usr)
    time = datetime.datetime.fromtimestamp(int(t)/1000.0) 
    if prev_time==-1:
        prev_time = time
    time_diff = time-prev_time
    print time_diff
    #timee = time.gmtime(int(t))
    if acc<200 and (prev_usr==-1 or prev_usr==userid):
        if prev_usr==-1:
            prev_usr = userid
            prev_time = time
        elif userid==prev_usr and act in activity_list:
            acc += 1
            val_list_x.append(float(x))
            val_list_y.append(float(y))
            val_list_z.append(float(z))            
    else:
        if acc==200:
        #set of transformations
            attribute = [0.0]*44
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
            attribute[30] = sum_x/len(val_list_x)
            attribute[31] = sum_y/len(val_list_y)
            attribute[32] = sum_z/len(val_list_z)
            attribute[43] = math.sqrt((sum_x**2)+(sum_y**2)+(sum_z**2))
            #print userid, attribute
            #write to transformed file
            tx_file.write(str(userid)+",")
            for attr in attribute:
                tx_file.write(str(attr)+",")
            tx_file.write("\n")
        raw_input()
        acc = 0
        val_list_x[:] = []
        val_list_y[:] = []
        val_list_z[:] = []
        prev_usr = userid
        prev_time = time
        val_list_x.append(float(x))
        val_list_y.append(float(y))
        val_list_z.append(float(z))