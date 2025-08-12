import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl
import time
import numpy as np

mpl.use('TkAgg')  # or use 'Agg' for non-GUI environments


class Animater:
    def __init__(self, data_generator, frames=100, xlim=(0, 1), ylim=(-1, 1),
                 xlabel='x', ylabel='u(x,t)', title='Wave Evolution'):
        """
        data_generator: function(frame_index) -> (x, u, k, E, pdf_x, pdf_y, t)
        frames: number of animation frames
        """
        self.data_generator = data_generator
        self.frames = frames
        self.xlim = xlim
        self.ylim = ylim
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title
        self.current_time = time.time()

        # Create figure with three subplots
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(1, 3, figsize=(15, 5))
        self.line1, = self.ax1.plot([], [], lw=1.5, label='u(x,t)')
        self.line2, = self.ax2.plot([], [], lw=1.5, label='E(k)')
        self.line3, = self.ax3.plot([], [], lw=1.5, label='PDF')
        self.time_text = self.ax1.text(0.02, 0.90, '', transform=self.ax1.transAxes)

        self._setup_plot()

    def _setup_plot(self):
        # Setup first subplot: u(x,t)
        self.ax1.set_xlim(*self.xlim)
        self.ax1.set_ylim(*self.ylim)
        self.ax1.set_xlabel(self.xlabel)
        self.ax1.set_ylabel(self.ylabel)
        self.ax1.set_title('Solution u(x,t)')
        self.ax1.legend()
        self.ax1.grid(True)

        # Setup second subplot: E(k)
        self.ax2.set_xlabel('k')
        self.ax2.set_ylabel('E(k)')
        self.ax2.set_title('Energy Spectrum')
        self.ax2.legend()
        self.ax2.grid(True)

        # Setup third subplot: PDF
        self.ax3.set_xlabel('u')
        self.ax3.set_ylabel('Probability Density')
        self.ax3.set_title('Probability Density Function')
        self.ax3.legend()
        self.ax3.grid(True)

        plt.tight_layout()

    def _init(self):
        self.line1.set_data([], [])
        self.line2.set_data([], [])
        self.line3.set_data([], [])
        self.time_text.set_text('')
        return self.line1, self.line2, self.line3, self.time_text

    def _update(self, i):
        x, u, k, E, pdf_x, pdf_y, t = self.data_generator(i)

        # Update u(x,t) plot
        self.line1.set_data(x, u)
        self.ax1.relim()
        self.ax1.autoscale_view()

        # Update E(k) plot
        self.line2.set_data(k, E)
        self.ax2.relim()
        self.ax2.autoscale_view()

        # Update PDF plot
        self.line3.set_data(pdf_x, pdf_y)
        self.ax3.relim()
        self.ax3.autoscale_view()

        self.time_text.set_text(f'time = {t:.3f}\n'
                                f'FPS = {1 / (time.time() - self.current_time):.0f}')
        self.current_time = time.time()
        return self.line1, self.line2, self.line3, self.time_text

    def start(self, interval=100):
        self.anim = animation.FuncAnimation(
            self.fig, self._update, init_func=self._init,
            frames=self._frame_gen(),
            interval=interval, blit=False, repeat=False, cache_frame_data=False
        )
        plt.show()

    def _frame_gen(self):
        i = 0
        while True:
            yield i
            i += 1