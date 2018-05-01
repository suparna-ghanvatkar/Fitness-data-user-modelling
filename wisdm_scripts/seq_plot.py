'''
Script to plot the activity frequent pattern sequence
'''
from activity_plot import datetime_range
import argparse
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('pat_file',type=str,help='path to freq patt txt file')
args = parser.parse_args()

act_legend = {'S':'Sitting','D':'Standing','W':'Walking','R':'Stairs','J':'Jogging','L':'LyingDown','g':'gap'}
act_color = {'S':'Red','D':'Green','W':'Brown','R':'Blue','J':'Pink','L':'Purple','g':'Black'}
f = open(args.pat_file,'r')
q = 0
plt.figure()
for line in f:
    contents = line.split(' ')
    pat = contents[0]
    #creating a table out of the pattern:
    strt_time = timedelta(0)
    data = []
    for i,act in enumerate(pat):
        data.append([i+1, i+2, act_legend[act], act_color[act]])
    df = pd.DataFrame(data, columns=['times','timee','act','color'])
    lines = []
    x = []
    y = []
    colors = []
    for i,row in df.iterrows():
        x.append(row['times'])
        x.append(row['timee'])
        y.append(10-q)
        y.append(10-q)
        colors.append(row['color'])
        colors.append(row['color'])
        lines.append(([i+1,i+2],[10-q, 10-q],row['color']))
    plt.scatter(x,y, color=colors)
    for line in lines:
        print line
        plt.plot(line[0], line[1], line[2])
    q += 1
plt.savefig(args.pat_file+'_plot.png',format='png',dpi=1200)
    #q += 1
