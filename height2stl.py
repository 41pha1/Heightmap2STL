#create stl file from .exr heightmap file

import os
import sys
import numpy as np
import cv2 as cv
import trimesh as tm
try:
    import mapbox_earcut as earcut
    HAS_EARCUT = True
except Exception:
    HAS_EARCUT = False
from trimesh.exchange.obj import export_obj

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

class QuadTreeNode:
    def __init__(self, x, y, width, height, depth):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.depth = depth

    def corners(self):
        return [
            (self.x, self.y),
            (self.x + self.width, self.y),
            (self.x, self.y + self.height),
            (self.x + self.width, self.y + self.height),
        ]

def detail_magnitude(heightmap):
    """Calculate the detail magnitude of a heightmap.
    heightmap: a 2d array of floats between 0 and 1.
    returns: a 2d array of floats between 0 and 1.
    """

    blured = cv.GaussianBlur(heightmap,(11,11),0) * 255

    gX = cv.Sobel(blured, ddepth=cv.CV_64F, dx=1, dy=0, ksize=5)
    gY = cv.Sobel(blured, ddepth=cv.CV_64F, dx=0, dy=1, ksize=5)

    gX = cv.convertScaleAbs(gX)
    gY = cv.convertScaleAbs(gY)

    return cv.addWeighted(gX, 0.5, gY, 0.5, 0)

def triangulate(vertices, poly):
    """Triangulate a polygon.
    poly: a list of indices of vertices in clockwise order.
    returns: a list of triangle indices.
    """
    
    # Guard against degenerate polygons
    if len(poly) < 3:
        return []

    #place an additional vertex in the center of the polygon.
    points = [vertices[i] for i in poly]
    center = (sum([p[0] for p in points]) // len(points), sum([p[1] for p in points]) // len(points))
    center_index = len(vertices)
    vertices.append(center)

    # Triangulate the polygon.
    triangles = []

    for i in range(len(poly)):
        triangles.append([poly[i], center_index, poly[(i+1) % len(poly)]])

    return triangles

def createFaces(vertices, polys):
    """Triangulate a list of polygons.
    vertices: a list of vertices.
    polys: a list of list containing the indices of the vertices of each polygon.
    returns: a list of triangle indices.
    """

    triangles = []

    for poly in polys:
        if len(poly) >= 3:
            triangles += triangulate(vertices, poly)

    return triangles

def getSourroundingVertices(vert_set, x, y, w, h):
    """Find all vertices in a rectangle.
    vertices: a dictionary of vertices.
    x, y, w, h: the rectangle to search (integers).
    returns: a list of indices of the vertices in the rectangle.
    """

    north, east, south, west = [], [], [], []

    for i in range(x, x + w):
        point = (i, y)

        if point in vert_set:
            north.append(vert_set[point])

    for i in range(y, y + h):
        point = (x + w, i)

        if point in vert_set:
            east.append(vert_set[point])

    for i in range(x + w, x, -1):
        point = (i, y + h)

        if point in vert_set:
            south.append(vert_set[point])

    for i in range(y + h, y, -1):
        point = (x, i)

        if point in vert_set:
            west.append(vert_set[point])

    return north, east, south, west

def createPolyFaces(vert_set, leafs):
    """Find all sourrounding vertices of a leaf and create a polygonal face from them.
    vert_set: a dictionary of vertices.
    leafs: a list of QuadTreeNodes.
    returns: a list of list containing the indices of the vertices of each face.
    """

    # TODO: This function is very slow. Optimize it by sorting the vertices and using a binary search to find them.

    polys = []

    for leaf in leafs:
        n,e,s,w = getSourroundingVertices(vert_set, leaf.x, leaf.y, leaf.width, leaf.height)
        polys.append(n+e+s+w)

    return polys


def _quad_max_normal_angle_deg(heightmap, n, aspect, z_scale, x, y, w, h):
    """
    Hybrid curvature metric (degrees):
    - max angle between the two triangle normals of the quad
    - max normal spread computed from finite differences at the four quad corners
    Returns the maximum of these two, making the metric sensitive to both fold and smooth curvature.
    """
    # Clamp to valid range
    x0, y0 = x, y
    x1, y1 = min(x + w, n), min(y + h, n)

    # Build 3D points for quad corners in world units consistent with createTerrain
    def P(ix, iy):
        return np.array([
            iy / n,                 # X-world maps from grid y
            aspect * (ix / n),      # Y-world maps from grid x scaled by aspect
            float(heightmap[iy, ix]) * z_scale
        ], dtype=np.float64)

    p00 = P(x0, y0)
    p10 = P(x1, y0)
    p01 = P(x0, y1)
    p11 = P(x1, y1)

    # Two triangles: (p00, p10, p11) and (p00, p11, p01)
    def tri_normal(a, b, c):
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm == 0.0:
            return np.array([0.0, 0.0, 0.0])
        return n / norm

    n1 = tri_normal(p00, p10, p11)
    n2 = tri_normal(p00, p11, p01)
    dotv = float(np.clip(np.dot(n1, n2), -1.0, 1.0))
    angle_rad = np.arccos(dotv) if -1.0 <= dotv <= 1.0 else 0.0
    tri_angle = float(np.degrees(angle_rad))

    # Corner normal spread via finite differences on heightmap
    def corner_normal(ix, iy):
        # central differences in grid space (dx, dy) ~ (1/n, 1/n)
        ix0 = max(ix - 1, 0); ix1 = min(ix + 1, n)
        iy0 = max(iy - 1, 0); iy1 = min(iy + 1, n)
        dzdx = (float(heightmap[iy, ix1]) - float(heightmap[iy, ix0])) * z_scale / ( (ix1 - ix0) / n if (ix1 - ix0) > 0 else 1.0 )
        dzdy = (float(heightmap[iy1, ix]) - float(heightmap[iy0, ix])) * z_scale / ( (iy1 - iy0) / n if (iy1 - iy0) > 0 else 1.0 )
        # normal in world scaling (x scaled by aspect)
        nx = -dzdx / aspect
        ny = -dzdy
        nvec = np.array([nx, ny, 1.0], dtype=np.float64)
        nrm = np.linalg.norm(nvec)
        return nvec / nrm if nrm > 0 else np.array([0.0, 0.0, 1.0])

    cn00 = corner_normal(x0, y0)
    cn10 = corner_normal(x1, y0)
    cn01 = corner_normal(x0, y1)
    cn11 = corner_normal(x1, y1)
    corners = [cn00, cn10, cn01, cn11]
    spread = 0.0
    for i in range(4):
        for j in range(i+1, 4):
            d = float(np.clip(np.dot(corners[i], corners[j]), -1.0, 1.0))
            a = np.degrees(np.arccos(d)) if -1.0 <= d <= 1.0 else 0.0
            if a > spread: spread = a

    return max(tri_angle, spread)

def subdivideAdaptiveAngle(heightmap, max_subdivisions, angle_threshold_deg, z_scale, aspect):
    """Subdivide adaptively based on the quad normal angle metric.
    heightmap: (n+1)x(n+1) float32 in [0,1]
    max_subdivisions: int, total quadtree depth
    angle_threshold_deg: float, subdivide if max triangle-normal angle of a quad exceeds this
    z_scale: float, consistent with createTerrain
    aspect: float, width/height of the input section
    returns: (vertices:list[(x,y)], triangles:list[[i,j,k]], (north,east,south,west))
    """
    n = 2 ** max_subdivisions

    # Root node covers whole [0..n]x[0..n]
    plane = [QuadTreeNode(0, 0, n, n, 0)]
    leafs = []
    vert_set = {}

    print("Subdividing plane {} times (angle metric)...".format(max_subdivisions))
    for _ in range(max_subdivisions):
        new_plane = []
        for node in plane:
            angle_deg = _quad_max_normal_angle_deg(heightmap, n, aspect, z_scale, node.x, node.y, node.width, node.height)
            if angle_deg > angle_threshold_deg and node.width > 1 and node.height > 1:
                hw = node.width // 2
                hh = node.height // 2
                new_plane.append(QuadTreeNode(node.x,           node.y,           hw, hh, node.depth + 1))
                new_plane.append(QuadTreeNode(node.x + hw,      node.y,           hw, hh, node.depth + 1))
                new_plane.append(QuadTreeNode(node.x,           node.y + hh,      hw, hh, node.depth + 1))
                new_plane.append(QuadTreeNode(node.x + hw,      node.y + hh,      hw, hh, node.depth + 1))
            else:
                # Add unique corners to vertex set
                for corner in node.corners():
                    if corner not in vert_set:
                        vert_set[corner] = len(vert_set)
                leafs.append(node)
        plane = new_plane

    # Remaining nodes are leaves
    leafs += plane
    # Ensure corners for the last-level leaves are present in the vertex set
    for node in plane:
        for corner in node.corners():
            if corner not in vert_set:
                vert_set[corner] = len(vert_set)

    print("Creating polygonal faces from {} nodes...".format(len(vert_set)))
    polys = createPolyFaces(vert_set, leafs)

    vertices = [None] * len(vert_set)
    for key in vert_set:
        vertices[vert_set[key]] = key

    print("Triangulating {} polygons...".format(len(polys)))
    triangles = createFaces(vertices, polys)

    print("Done.")
    return vertices, triangles, getSourroundingVertices(vert_set, 0, 0, n, n)

def createTerrain(heightmap, max_subdivisions, angle_threshold_deg, z_scale, ground_height, tile_i, tile_j, nx, ny):
    """Create a terrain volume from a heightmap.
    heightmap: a 2d array of floats between 0 and 1.
    max_subdivisions: the maximum number of times to subdivide the plane.
    angle_threshold_deg: subdivide when quad's max triangle-normal angle exceeds this (degrees).
    z_scale: the height of the terrain.
    ground_height: the thickness of the ground.
    """

    # Validate the inputs.
    assert max_subdivisions > 0
    assert angle_threshold_deg >= 0

    n = 2 ** max_subdivisions
    aspect = heightmap.shape[1] / heightmap.shape[0]
    heightmap = cv.resize(heightmap, (n + 1, n + 1), interpolation=cv.INTER_CUBIC)

    # Subdivide using angle metric
    vertices, triangles, (north, east, south, west) = subdivideAdaptiveAngle(heightmap, max_subdivisions, angle_threshold_deg, z_scale, aspect)

    # Adjust the vertices to the correct height.
    print("Adjusting vertices...")
    # Prepare UVs (global 0..1 across full nx x ny tiling)
    uvs = []
    for i in range(len(vertices)):
        gx = vertices[i][0]
        gy = vertices[i][1]
        # World positions aligned to image axes: x across width, y across height
        wx = (gx / n) * aspect
        wy = (gy / n)
        wz = float(heightmap[gy, gx]) * z_scale
        vertices[i] = [wx, wy, wz]
        # UVs: map to global image [0,1] using tile indices
        u = (tile_i + (gx / n)) / nx
        v = (tile_j + (gy / n)) / ny
        uvs.append([u, v])
    
    # Create the sides of the terrain.
    sides = [north + east[:1], east + south[:1], south + west[:1], west + north[:1]]
    edge = []
    for side in sides:
        for i in range(len(side)):
            vertex = vertices[side[i]].copy()
            vertex[2] = ground_height
            vertices.append(vertex)
            edge.append(len(vertices) - 1)
            # Duplicate UV for side vertex from the top vertex
            uvs.append(uvs[side[i]])

            if i > 0:
                triangles.append([side[i], len(vertices)-1, side[i-1]])
                triangles.append([len(vertices) - 2, side[i-1], len(vertices)-1])

    # Create the bottom of the terrain by triangulating the boundary polygon using earcut if available
    if HAS_EARCUT and len(edge) >= 3:
        # Build 2D polygon (x,y) from bottom edge vertices
        data = np.array([[vertices[idx][0], vertices[idx][1]] for idx in edge], dtype=np.float32)
        ring_ends = np.array([data.shape[0]], dtype=np.uint32)
        idxs = earcut.triangulate_float32(data, ring_ends)
        for t in range(0, len(idxs), 3):
            a = edge[int(idxs[t+0])]
            b = edge[int(idxs[t+1])]
            c = edge[int(idxs[t+2])]
            triangles.append([a, b, c])
    else:
        # Fallback: simple fan from average point
        cx = sum(vertices[i][0] for i in edge) / len(edge)
        cy = sum(vertices[i][1] for i in edge) / len(edge)
        cidx = len(vertices)
        vertices.append([cx, cy, ground_height])
        uvs.append([(tile_i + 0.5) / nx, (tile_j + 0.5) / ny])
        for i in range(len(edge)):
            triangles.append([edge[i], edge[(i+1) % len(edge)], cidx])

    print("Exporting meshes...")
    mesh = tm.Trimesh(vertices=np.array(vertices), faces=np.array(triangles), process=False)
    try:
        # Attach UVs for formats that support them (e.g., OBJ)
        mesh.visual = tm.visual.TextureVisuals(uv=np.array(uvs, dtype=np.float64))
    except Exception as e:
        print(f"Warning: failed to attach UVs: {e}")
    return mesh
    
def printHelp():
    print("Usage: python height2stl.py <input_file> [options]")
    print("Options:")
    print("  -n <nx = 1> <ny = 1>: the number of terrain meshes to create in the x and y directions.")
    print("  -s <max_subdivisions = 10>: the maximum number of times to subdivide the plane.")
    print("  -t <angle_threshold_deg = 5.0>: maximum quad normal angle before subdividing (degrees).")
    print("  -z <z_scale = 1>: the height of the terrain.")
    print("  -g <ground_height = 0>: the base elevation of the ground.")
    print("  -o <output_file>: the output file to write to. If not specified, the input file name will be used.")
    print("  -h: print this help message.")

def parseArgs(args):
    """Parse the command line arguments."""

    nx = 1
    ny = 1
    max_subdivisions = 10
    threshold = 5.0
    z_scale = 1
    ground_height = 0
    input_file = None
    output_file = None

    if len(args) == 0 or args[0] == "-h":
        printHelp()
        sys.exit(0)

    input_file = args[0]
    output_file = os.path.splitext(input_file)[0]

    i = 1
    while i < len(args):
        if args[i] == "-n":
            nx = int(args[i+1])
            ny = int(args[i+2])
            i += 3
        elif args[i] == "-s":
            max_subdivisions = int(args[i+1])
            i += 2
        elif args[i] == "-t":
            threshold = float(args[i+1])
            i += 2
        elif args[i] == "-z":
            z_scale = float(args[i+1])
            i += 2
        elif args[i] == "-g":
            ground_height = float(args[i+1])
            i += 2
        elif args[i] == "-o":
            output_file = args[i+1]
            i += 2
        elif args[i] == "-h":
            printHelp()
            sys.exit(0)
        else:
            print("Error: unknown option '{}'.".format(args[i]))
            printHelp()
            sys.exit(1)

    return nx, ny, max_subdivisions, threshold, z_scale, ground_height, input_file, output_file

def main():
    nx, ny, max_subdivisions, threshold, z_scale, ground_height, input_file, output_file = parseArgs(sys.argv[1:])
    output_dir = os.path.splitext(output_file)[0]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Reading heightmap...")
    heightmap = cv.imread(input_file, cv.IMREAD_GRAYSCALE | cv.IMREAD_ANYDEPTH)
    heightmap = np.array(heightmap, dtype=np.float32)
    heightmap /= max(heightmap.flatten())

    w = heightmap.shape[1] // nx
    h = heightmap.shape[0] // ny

    for i in range(nx):
        for j in range(ny):
            print("Creating terrain for ({}, {})...".format(i, j))
            heightmap_section = heightmap[j*h:(j+1)*h, i*w:(i+1)*w]
            terrain_mesh = createTerrain(heightmap_section, max_subdivisions, threshold, z_scale, ground_height, i, j, nx, ny)
            # Export STL (geometry only)
            stl_path = os.path.join(output_dir, output_file + "_{}_{}.stl".format(i, j))
            terrain_mesh.export(stl_path)
            # Also export OBJ with UVs for texturing workflows
            # Rotate Z-up (internal) to Y-up (OBJ common convention): (x, y, z) -> (x, z, y)
            obj_mesh = terrain_mesh.copy()
            v = obj_mesh.vertices
            obj_mesh.vertices = np.stack([v[:,0], v[:,2], v[:,1]], axis=1)
            obj_path = os.path.join(output_dir, output_file + "_{}_{}.obj".format(i, j))
            with open(obj_path, 'w') as f:
                f.write(export_obj(obj_mesh, include_texture=True))

if __name__ == "__main__":
    main()