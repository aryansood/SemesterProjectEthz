import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
import matplotlib.pyplot as plt

def save_to_mp4(images, path_to_save):
    frames = []
    fig = plt.figure()
    for i in range(0, images.shape[0]):
        frames.append([plt.imshow(images[i], animated=True)])
    ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True, repeat_delay=1000)
    writer = FFMpegWriter(fps=20, bitrate=1800)
    ani.save(path_to_save, writer=writer, dpi=300)
    plt.close(fig)