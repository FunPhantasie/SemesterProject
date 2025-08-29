import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl

mpl.use('TkAgg')
path = "Explicit_Solution"
files = sorted([f for f in os.listdir(path) if f.endswith(".png")])

fig, ax = plt.subplots(figsize=(12, 9))
ax.axis("off")  # remove axes

# Remove whitespace around the image
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

# Match the figure size to the image dimensions
img = plt.imread(os.path.join(path, files[0]))
im = ax.imshow(img, aspect="auto")


# add a text element for the time counter
time_text = ax.text(
    0.02, 0.95, "", color="white", fontsize=14,
    ha="left", va="top", transform=ax.transAxes,
    bbox=dict(facecolor="black", alpha=0.5, edgecolor="none")
)

def update(i):
    frame_file = files[i]
    im.set_array(plt.imread(os.path.join(path, frame_file)))
    # Extract time from filename or compute from frame index
    time_text.set_text(f"t = {i}")
    return [im, time_text]

ani = animation.FuncAnimation(fig, update, frames=len(files), interval=100, blit=True)

plt.show()
# To save:
# ani.save("simulation.mp4", writer="ffmpeg", fps=10)
# ani.save("simulation.gif", writer="pillow", fps=10)
