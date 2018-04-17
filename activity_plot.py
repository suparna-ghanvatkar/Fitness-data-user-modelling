import argparse
import pickle
import datetime
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
import random
import matplotlib

matplotlib.use('Agg')

def datetime_range(start,end,delta):
    current = start
    while current<end:
        yield current
        current += delta

def plot_timeline(data, outpath):
    series = ['Walking','Jogging','Stairs','Standing','Sitting','LyingDown']
    x = []
    y = []
    colors = []
    lines = []
    base_colors = ['red','green','blue','black']
    color_counter = 0
    plt.figure(figsize=(100,6))
    plt.ylim((-1,len(series)))
    start_time = data.iloc[0]['start']
    end_time = data.iloc[-1]['end']
    for i,row in data.iterrows():
        #print row
        activity = row['activity']
        times = row['start'].time()
        timee = row['end'].time()
        lines.append(([times,activity],[timee,activity]))
        x.append(times)
        x.append(timee)
        y.append(activity)
        y.append(activity)
        colors.append(base_colors[color_counter%4])
        colors.append(base_colors[color_counter%4])
        color_counter += 1

#    plt.plot(x,y,'ro')
#    for line in lines:
#        plt.plot(line[0],line[1],'k-')
    yticks = [dt.strftime('%H:%M:%s') for dt in datetime_range(start_time-timedelta(minutes=1), end_time, timedelta(minutes=1))]
    xpos = np.linspace(0,1,len(yticks))
    #plt.xticks(np.arange(len(yticks)),yticks, rotation=65)
    #print plt.xticks()
    plt.grid()
    plt.scatter(x,y, color=colors)
    return plt.savefig(outpath, format = 'svg', dpi=1200)

parser = argparse.ArgumentParser()
parser.add_argument('daywise_act', type=str, help='path to day wise user wise activity data')
parser.add_argument('outpath', type=str, help='path to output svg file')
args = parser.parse_args()

data = pickle.load(open(args.daywise_act,'rb'))
users = data.user.unique()
print data.dtypes
print users
usr = random.choice(users)
usr = 1320
print usr
user_data = data.loc[data.user==usr]
dates = user_data.date.unique()
dt = random.choice(dates)
dt = datetime.date(year=2013, month=10, day=2)
print dt
#print user_data
plot_data = user_data.loc[user_data.date==dt][['activity','start','end']]
print plot_data
plot_timeline(plot_data,args.outpath)
