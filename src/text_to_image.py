
import numpy as np
import matplotlib.pyplot as plt
import sys


if len(sys.argv) != 3:
    print("Usage: python src/image_to_text.py <input.txt> <output.png>")
    sys.exit(1)


# info for what file and where to store it.
text_file = sys.argv[1]
result_file = sys.argv[2]


# Load converted text file
with open(text_file, "r") as file:
    width, height = map(int, file.readline().split())

    pixels = np.loadtxt(file)

# build the image
converted_array = pixels.reshape((height, width))

# Save image in the folder
output_image = result_file

plt.imsave(
    output_image,
    converted_array,
    cmap="gray",
    vmin=0,
    vmax=255
)

print(f"Image saved to {output_image}")

