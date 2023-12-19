#create stl file from .exr heightmap file

import os
import sys
import numpy as np
import cv2 as cv
import scipy.interpolate as irp
from PIL import Image
import trimesh 
import random

nx = 3
ny = 2

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

class Node:
    children = []
    parent = None
    done = False

    def __init__(self, parent):
        self.parent = parent
    
    def get_children(self):
        return self.children
    
    def get_parent(self):
        return self.parent

    def subdivide(self):
        if self.children:
            return self.children

        self.children = []
        self.children.append(Node(self))
        self.children.append(Node(self))
        self.children.append(Node(self))
        self.children.append(Node(self))

        return self.children

    def iterate(self, x, y, w, h, func, *args):
        if self.children:
            self.children[0].iterate(x, y, w / 2, h / 2, func, *args)
            self.children[1].iterate(x + w / 2, y, w / 2, h / 2, func, *args)
            self.children[2].iterate(x, y + h / 2, w / 2, h / 2, func, *args)
            self.children[3].iterate(x + w / 2, y + h / 2, w / 2, h / 2, func, *args)
        else:
            func(self, x, y, w, h, *args)

    def draw(self, img, x, y, w, h, stroke_width):
        if self.children:
            self.children[0].draw(img, x, y, w // 2, h // 2, stroke_width // 1.5)
            self.children[1].draw(img, x + w // 2, y, w // 2, h // 2, stroke_width // 1.5)
            self.children[2].draw(img, x, y + h // 2, w // 2, h // 2, stroke_width // 1.5)
            self.children[3].draw(img, x + w // 2, y + h // 2, w // 2, h // 2, stroke_width // 1.5)
        
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), max(2,int(stroke_width))) 

def subdivide_node(node, x, y, w, h, threshold, cumsum):
    if node.done:
        return 
    
    detail = cumsum(x + w, y + h) - cumsum(x, y + h) - cumsum(x + w, y) + cumsum(x, y)

    if detail > threshold:
        node.subdivide()
    else:
        node.done = True


class QuadTree:
    def __init__(self):
        self.root = Node(None)

    def subdivide(self, cumsum, w, h, threshold):
        self.root.iterate(0, 0, w, h, subdivide_node, threshold, cumsum)

    def draw (self, img):
        self.root.draw(img, 0, 0, img.shape[1], img.shape[0], 10)

def accumulated_sum(heightmap):
    acc_sum = np.cumsum(heightmap, axis=1)
    acc_sum = np.cumsum(acc_sum, axis=0)

    return acc_sum

def adjust_vertices(vertices, detail, inter, n):
    v = 4
    for i in range(n):
        for j in range(n):
            dx = i * detail.shape[0] // n
            dy = j * detail.shape[1] // n

            # magnitude = max(0, min(1 / (detail[dx, dy]+ 1e-10), 1)) * 0.001
            # vertices[v][0] += random.uniform(-magnitude, magnitude)
            # vertices[v][1] += random.uniform(-magnitude, magnitude) 
            vertices[v][2] = 0.005 + inter(dy, dx) / 800

            v += 1

    return vertices

def create_stl(tile):
    gray = cv.cvtColor(tile, cv.COLOR_BGR2GRAY) * 255
    detail = detail_magnitude(gray)

    x = np.linspace(0, gray.shape[1], gray.shape[1])
    y = np.linspace(0, gray.shape[0], gray.shape[0])

    inter = irp.interp2d(x, y, gray, kind='cubic')

    n = 1000
    vertices = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
    ]

    faces = [
        [0, 2, 1],
        [2, 3, 1],
    ]

    v = 3
    f = 1

    for i in range(n):
        for j in range(n):
            ty = tile.shape[1] * j / (n-1)
            tx = tile.shape[0] * i / (n-1)
            vertices.append([i / (n-1), j / (n-1), 1])
            v += 1

            if j == 0 and i != 0:
                faces.append([v-n, 0, v])
                if i == n-1:
                    faces.append([0, 1, v])

            if j == n-1 and i != 0:
                faces.append([v, 2, v-n])
                if i == n-1:
                    faces.append([3, 2, v])

            if i == 0 and j != 0:
                faces.append([v, 0, v - 1])
                if j == n-1:
                    faces.append([2, 0, v])

            if i == n-1 and j != 0:
                faces.append([v-1, 1, v])
                if j == n-1:
                    faces.append([1, 3, v])

            if j != 0 and i != 0:
                faces.append([v, v - n, v - 1])
                faces.append([v-1, v - n, v - n -1])

    vertices = adjust_vertices(vertices, detail, inter, n)
    faces = np.array(faces)

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    return mesh 
    
def detail_magnitude(gray):

    gray = cv.GaussianBlur(gray,(21,21),0)

    gX = cv.Sobel(gray, ddepth=cv.CV_64F, dx=1, dy=0, ksize=5)
    gY = cv.Sobel(gray, ddepth=cv.CV_64F, dx=0, dy=1, ksize=5)

    gX = cv.convertScaleAbs(gX)
    gY = cv.convertScaleAbs(gY)

    combined = cv.addWeighted(gX, 0.5, gY, 0.5, 0)
    #vector_field = np.dstack((128 + gX, 128 + gY, np.zeros_like(gX)))
    #cv.imwrite("detail_magnitude.png", vector_field)

    return combined

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
    edges = detail_magnitude(heightmap * 255)

    cumsum_file = input_file.replace(".exr", "_cumsum.npy")
    if not os.path.isfile(cumsum_file):
        cumsum = accumulated_sum(edges)
        cumsum = cumsum.astype(np.float32)
        np.save(cumsum_file, cumsum)
    else:
        cumsum = np.load(cumsum_file)

    cumsum = np.array(cumsum, dtype=np.float32)
    cumsum = cv.cvtColor(cumsum, cv.COLOR_BGR2GRAY)

    # print("Creating Interpolation...")
    # x = np.linspace(0, cumsum.shape[1], cumsum.shape[1])
    # y = np.linspace(0, cumsum.shape[0], cumsum.shape[0])

    # cumsum_inter = irp.interp2d(x, y, cumsum, kind='cubic')

    # print("Creating Quadtree...")
    # quadtree = QuadTree()

    # for i in range(1, 12):
    #     print("Iteration: {}".format(i))
    #     quadtree.subdivide(cumsum_inter, cumsum.shape[1], cumsum.shape[0], 2000)

    # print("Creating STL...")
    # quadtree.draw(edges)
    # cv.imwrite("quadtree.png", edges)



    # exit()
    
    # normalize heightmap to [0, 1]
    heightmap = heightmap - np.min(heightmap)
    heightmap = heightmap / np.max(heightmap)

    # split heightmap into nx * ny tiles
    h, w, _ = heightmap.shape
    tile_h = h // ny
    tile_w = w // nx

    for i in range(ny):
        for j in range(nx):
            print(i, j)
            mesh = create_stl(heightmap[i * tile_h : (i + 1) * tile_h, j * tile_w : (j + 1) * tile_w])
            mesh.export(os.path.join(output_dir, "tile_{}_{}.stl".format(i, j)))


if __name__ == "__main__":
    main()