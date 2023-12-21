#create stl file from .exr heightmap file

import os
import sys
import numpy as np
import cv2 as cv
import trimesh as tm
from meshBuilder import createTerrain

nx = 1
ny = 1

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
    heightmap = cv.imread(input_file, cv.IMREAD_GRAYSCALE | cv.IMREAD_ANYDEPTH)
    heightmap = np.array(heightmap, dtype=np.float32)
    heightmap /= max(heightmap.flatten())
    

    w = heightmap.shape[1] // nx
    h = heightmap.shape[0] // ny

    for i in range(nx):
        for j in range(ny):
            print("Creating terrain for ({}, {})...".format(i, j))
            heightmap_section = heightmap[j*h:(j+1)*h, i*w:(i+1)*w]
            terrain_mesh = createTerrain(heightmap_section, 12, 0.0005, 0.4, -0.01)
            terrain_mesh.export(os.path.join(output_dir, "terrain_{}_{}.stl".format(i, j)))

if __name__ == "__main__":
    main()