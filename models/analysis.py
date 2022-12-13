from czifile import CziFile
from czifile import imread as cziRead

from time import time

from .frame import Frame
from .imageData import ImageData


class Analysis(object):
    pathFile: str
    minClustervolume = 500
    minRadiusCell = 7.5
    approximateAmountOfNuclei = 80

    imageData: ImageData
    frames: list[Frame]

    def __init__(self, pathFile: str):
        self.pathFile = pathFile
        timeBefLoadImage = time()
        self.imageData = ImageData(
            cziRead(pathFile), CziFile(pathFile).metadata(False))
        print("[INFO] LoadImage: " + str(time() - timeBefLoadImage))
        self.frames = list(map(self.__genereteFrame, range(self.imageData.sizes.z-1)))

    def __genereteFrame(self, sliceId):
        return Frame(sliceId, self.imageData.getVoxelFrame(sliceId))
