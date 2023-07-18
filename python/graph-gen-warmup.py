import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
import sys
import datetime as dt

def read_data(file): 
    step=[]
    median=[]
    stability=[]
    line = file.readline()
    while True:
        line = file.readline()
        if not line:
                break
        line = line.replace('\n', '')
        line = line.replace(' ', '')
        line = line.split(sep="|")
        step.append(int(line[0]))
        median.append(int(line[1]))
        stability.append(float(line[2]))
    return step, median, stability


def main() :
    file = open(sys.argv[2])
    step, median, stability = read_data(file)

    fig, ax1 = plt.subplots(1,1)

    # formatter = ticker.ScalarFormatter()
    # formatter.set_scientific(False)
    # ax1.yaxis.set_major_formatter(formatter)

    ax2 = ax1.twinx()
    ax1.plot(median, color='b', label="median value")
    ax1.set_ylabel(f'{sys.argv[4]}')
    ax1.plot(0, color='y', label="(med-min)/min")
    ax2.plot(stability, color='y', label="(med-min)/min")
    ax2.set_ylabel('Stability')
    ax1.legend()
    ax2.yaxis.set_major_formatter(ticker.PercentFormatter())
    ax1.set_xlabel("Steps")
    ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax1.set_ylim(ymin=0)
    ax1.set_title(f"Warmup calibration for a {sys.argv[1]}x{sys.argv[1]} matrix")
    plt.tight_layout()
    plt.savefig(sys.argv[3])

if __name__ == '__main__':
    main()

