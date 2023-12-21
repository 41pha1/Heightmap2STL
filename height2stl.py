#create stl file from .exr heightmap file

import os
import sys
import numpy as np
import cv2 as cv
import scipy.interpolate as irp
from PIL import Image
import random
from meshBuilder import createTerrain

nx = 3
ny = 2

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

def main():
    if len(sys.argv) < 2:
        print("Usage: python convert.py <input.exr>")
        return

    input_file = sys.argv[1]
    output_dir = os.path.splitext(input_file)[0]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Reading heightmap and detecting features...")
    heightmap = cv.imread(input_file, cv.IMREAD_ANYCOLOR | cv.IMREAD_ANYDEPTH)
    heightmap = np.array(heightmap, dtype=np.float32)
    heightmap = cv.cvtColor(heightmap, cv.COLOR_BGR2GRAY)

    createTerrain(heightmap, 8, 0.01, 0.2, -0.1)

if __name__ == "__main__":
    main()