import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import radon, iradon
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import matplotlib as mpl
mpl.use('TkAgg')

# 1. Kreis-Phantom erzeugen
def create_circle_phantom(N, radius_fraction=0.3):
    x = np.linspace(-1, 1, N)
    y = np.linspace(-1, 1, N)
    xx, yy = np.meshgrid(x, y)
    circle = (xx**2 + yy**2) <= (radius_fraction**2)
    return circle.astype(float)

N = 256
phantom = create_circle_phantom(N)

# 2. Projektionen berechnen (Radon-Transformation)
theta = np.linspace(0., 180., max(N//2, 1), endpoint=False)
sinogram = radon(phantom, theta=theta, circle=True)

# 3. Ungefilterte Rückprojektion (Backprojection)
reconstruction_unfiltered = iradon(sinogram, theta=theta, filter_name=None, circle=True)

# 4. Gefilterte Rückprojektion (FBP)
reconstruction_filtered = iradon(sinogram, theta=theta, filter_name='ramp', circle=True)

# 5. 2D-Fourier-Rücktransformation (idealisierter Vergleich)
fft2_image = fftshift(fft2(phantom))
ifft2_image = np.abs(ifft2(ifftshift(fft2_image)))

# 6. Visualisierung
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

axs[0, 0].imshow(phantom, cmap='gray')
axs[0, 0].set_title('Originalbild (Kreis)')
axs[0, 0].axis('off')

axs[0, 1].imshow(sinogram, cmap='gray', aspect='auto')
axs[0, 1].set_title('Sinogramm (Radonraum)')
axs[0, 1].set_xlabel('Winkel (θ)')
axs[0, 1].set_ylabel('Detektorposition (r)')

axs[0, 2].imshow(reconstruction_unfiltered, cmap='gray')
axs[0, 2].set_title('Ungefilterte Rückprojektion')
axs[0, 2].axis('off')

axs[1, 0].imshow(reconstruction_filtered, cmap='gray')
axs[1, 0].set_title('Gefilterte Rückprojektion (FBP)')
axs[1, 0].axis('off')

axs[1, 1].imshow(ifft2_image, cmap='gray')
axs[1, 1].set_title('Rücktransformation aus FFT2')
axs[1, 1].axis('off')

axs[1, 2].axis('off')  # leer

plt.tight_layout()
plt.show()
