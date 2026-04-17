import numpy as np
from shapely.geometry import Point, Polygon, box
from shapely.affinity import scale, rotate, translate


class PrimitiveShapes:
    def __init__(self, canvas_size=256):
        self.canvas_size = canvas_size

    def circle(self, center=None, radius_ratio=0.25):
        center = center or (self.canvas_size / 2, self.canvas_size / 2)
        radius = self.canvas_size * radius_ratio

        return Point(center).buffer(radius)

    def square(self, center=None, side_ratio=0.5):
        side = self.canvas_size * side_ratio
        center = center or (self.canvas_size / 2, self.canvas_size / 2)
        x0 = center[0] - side / 2
        y0 = center[1] - side / 2

        return box(x0, y0, x0 + side, y0 + side)

    def rectangle(self, top_left=None, bottom_right=None):
        if top_left is None:
            h, w = self.canvas_size, self.canvas_size
            top_left = ( int(h / 4), int(1.5 * w / 4))
            bottom_right = (int(3 * h / 4), int(2.5 * w / 4) )

        return box(top_left[0], top_left[1], bottom_right[0], bottom_right[1])

    def triangle(self, points=None):
        if points is None:
            h, w = self.canvas_size, self.canvas_size
            points = [
                (w / 2, h / 4),
                (w / 4, 3 * h / 4),
                (3 * w / 4, 3 * h / 4)
            ]

        return Polygon(points)

    def ellipse(self, center=None, axes_ratio=(0.3, 0.2)):
        center = center or (self.canvas_size / 2, self.canvas_size / 2)
        rx = self.canvas_size * axes_ratio[0]
        ry = self.canvas_size * axes_ratio[1]
        circ = Point(center).buffer(1.0, resolution=64)
        ell = scale(circ, rx, ry)

        return ell

    def diamond(self, center=None, size_ratio=0.5):
        center = center or (self.canvas_size / 2, self.canvas_size / 2)
        half = self.canvas_size * size_ratio / 2
        points = [
            (center[0], center[1] - half),
            (center[0] + half, center[1]),
            (center[0], center[1] + half),
            (center[0] - half, center[1])
        ]

        return Polygon(points)
    
    def pentagon(self, center=None, radius_ratio=0.4):
        center = center or (self.canvas_size / 2, self.canvas_size / 2)
        radius = self.canvas_size * radius_ratio
        cx, cy = center
        points = []

        angle_offset = np.pi - np.pi / 10  # rotate to make bottom edge flat
        for i in range(5):
            angle = 2 * np.pi * i / 5 + angle_offset
            x = cx + radius * np.cos(angle)
            y = cy + radius * np.sin(angle)
            points.append((x, y))

        return Polygon(points)
    
    def sector(self, center=None, radius=80, start_angle_deg=0, end_angle_deg=30, resolution=100):
        """
        Create a circular sector using shapely
        - center: (x, y)
        - radius: scalar radius
        - start_angle_deg, end_angle_deg: in degrees, counter-clockwise
        - resolution: number of points to sample along arc
        """
        center = center or (self.canvas_size / 2, self.canvas_size / 2)
        cx, cy = center
        start_rad = np.deg2rad(start_angle_deg)
        end_rad = np.deg2rad(end_angle_deg)
        
        if end_rad < start_rad:
            end_rad += 2 * np.pi

        arc_thetas = np.linspace(start_rad, end_rad, resolution)
        arc_points = [(cx + radius * np.cos(t), cy + radius * np.sin(t)) for t in arc_thetas]

        points = [center] + arc_points + [center]

        return Polygon(points)