import pickle
import matplotlib.pyplot as plt

def plot_timeline(data, outpath):
    series = ['Walking','Jogging','Stairs','Standing','Sitting','LyingDown']
    x = []
    y = []
    for activity, times, timee in data:
        x.append(times)
        x.append(timee)
        y.append(activity)
        y.append(activity)

    plt.figure(figsize=(24,6))
    plt.ylim((-1,len(series)))
    plt.scatter(x,y)
    return plt.savefig(outpath, format = 'svg', dpi=1200)

