from numpy import memmap, ndarray, shape
from services.cziservice import getDistanceMetaData

from sizes import ValuesAxis


class ImageData(object):
    #img: ndarray | memmap
    #distances: ValuesAxis[float]
    #sizes: ValuesAxis[int]
    #pixelRatio: float
    #slicePixelRatio: float

    def __init__(self, img: ndarray, metadata: dict):
        self.img = img
        self.sizes = ValuesAxis[int](shape(img)[4],
                                     shape(img)[5],
                                     shape(img)[3])
        self.distances = ValuesAxis[float](getDistanceMetaData(metadata, "X") * 10**7,
                                           getDistanceMetaData(metadata, "Y") * 10**7,
                                           getDistanceMetaData(metadata, "Z") * 10**6)
        self.pixelRatio = self.distances.x * self.distances.y
        self.slicePixelRatio = self.distances.z / self.distances.x
        

    def getVoxelFrame(self, frame: int) -> ndarray:
        return self.img[0, frame, 0, :, :, :, 0]
