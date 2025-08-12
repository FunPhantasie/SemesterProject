
import numpy as np
from animationstudio_module import *
# This is a sample Python script.

# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def example_generator(i):
    # Simulate 100 particles moving in a circle
    theta = np.linspace(0, 2*np.pi, 100)
    r = 0.5 + 0.1*np.sin(i * 0.1)  # radius oscillates
    x = r * np.cos(theta + i*0.05)
    y = r * np.sin(theta + i*0.05)
    t = i * 0.1
    return x, y, t




def print_hi(name):
    anim = AnimatedScatter(
        data_generator=example_generator,
        frames=200,
        xlim=(-1, 1),
        ylim=(-1, 1),
        xlabel='x',
        ylabel='y',
        title='Circle Animation'
    )

    anim.start()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
