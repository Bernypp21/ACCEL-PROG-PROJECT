
from PIL import Image
import numpy as np
import sys

def convert_image(input_file, output_file):
    # get image
    img = Image.open(input_file)

    # turn it grayscale
    img = img.convert("L")

    #set it in a NumPy array
    pixels = np.array(img, dtype=np.uint8)

    #image dimensions
    height, width = pixels.shape

    # save dimensions and pixel values
    with open(output_file, "w") as file:
        file.write(f"{width} {height}\n")

        np.savetxt(file, pixels, fmt="%d")

    print(f"Image converted successfully!")
    print(f"Width: {width}")
    print(f"Height: {height}")
    print(f"Output: {output_file}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python src/image_to_text.py <input.png> <output.txt>")
        sys.exit(1)

    convert_image(sys.argv[1], sys.argv[2])
