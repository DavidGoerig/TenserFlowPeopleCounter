#!/usr/bin/python3

import sys
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from matplotlib import style
style.use('ggplot')

xs = []
ys = []
fig = plt.figure()
axs = fig.add_subplot(1, 1, 1)


def graph_labels():
    plt.title("Fréquentation horaire de la pièce")
    plt.xlabel('Horaire')
    plt.ylabel('Nombre de personnes')


def graph_animate(fig, xs, ys):
    data = open('../created_data/data.txt', 'r').read()
    lines = data.split('\n')

    for line in lines:
        if len(line) > 1:
            x, y = line.split('_')
            xs.append(x.split(' ')[1].split('.')[0])
            ys.append(float(y))
    xs, ys = xs[-10:], ys[-10:]
    axs.clear()
    axs.plot(xs, ys, '-o')
    graph_labels()


def main():
    fig.canvas.set_window_title('Graph Window')
    _anim = animation.FuncAnimation(fig, graph_animate, fargs=(xs, ys), interval=1000)
    plt.show()


if __name__ == "__main__":
    main()
