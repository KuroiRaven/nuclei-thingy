from numpy import float16


class Voxel:
    x: int
    y: int
    z: float16

    def __init__(self, x: int, y: int, z: float16):
        self.x = x
        self.y = y
        self.z = z

    def __str__(self) -> str:
        return "("+self.x+", " + self.y + ", " + self.z+")"
