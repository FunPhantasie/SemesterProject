import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl
mpl.use('TkAgg')  # or use 'Agg' for non-GUI environments

class AnimatedScatter:
    def __init__(self, data_generator, frames=100, xlim=(0, 1), ylim=(-1, 1),
                 xlabel='x', ylabel='y', title='Animated Scatter'):
        """
        data_generator: function(frame_index) -> (x, y, t)
        frames: number of animation frames
        """
        self.data_generator = data_generator
        self.frames = frames
        self.xlim = xlim
        self.ylim = ylim
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title

        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [], lw=1.5)
        self.time_text = self.ax.text(0.02, 0.95, '', transform=self.ax.transAxes)

        self._setup_plot()

    def _setup_plot(self):
        self.ax.set_xlim(*self.xlim)
        self.ax.set_ylim(*self.ylim)
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)
        self.ax.set_title(self.title)

    def _init(self):
        self.line.set_data([], [])
        self.time_text.set_text('')
        return self.line, self.time_text

    def _update(self, i):
        x, y, t = self.data_generator(i)
        self.line.set_data(x, y)
        self.ax.relim()  # Recalculate limits
        self.ax.autoscale_view()  # Rescale axes to fit data
        self.time_text.set_text(f'time = {t:.2f}')
        return self.line, self.time_text

    def start(self, interval=100):
        self.anim = animation.FuncAnimation(
            self.fig, self._update, init_func=self._init,
            frames=self._frame_gen(),  # Infinite generator
            interval=interval, blit=False, repeat=False, cache_frame_data=False
        )
        plt.show()

    def _frame_gen(self):
        i = 0
        while True:
            yield i
            i += 1