from ..services.cziservice import getDistanceMetaData
from numpy import ndarray, memmap, shape
from sizes import ValuesAxis


class ImageData(object):
    img: ndarray | memmap
    distances: ValuesAxis[float]
    sizes: ValuesAxis[int]
    pixelRatio: float
    slicePixelRatio: float

    def __init__(self, img: ndarray | memmap, metadata: dict):
        self.img = img
        self.sizes = ValuesAxis[int](shape(img)[4],
                                     shape(img)[5],
                                     shape(img)[3])
        self.distances = ValuesAxis[float](getDistanceMetaData(metadata, "X") * 10**7,
                                           getDistanceMetaData(metadata, "Y") * 10**7,
                                           getDistanceMetaData(metadata, "Z") * 10**6)
        self.pixelRatio = self.distanceX * self.distanceY
        self.slicePixelRatio = self.pixelRatio * self.distanceZ

    def getVoxelFrame(self, frame: int) -> ndarray:
        return self.img[0, frame, 0, :, :, :, 0]
