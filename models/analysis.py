from imageData import ImageData
from frame import Frame
from czifile import CziFile, imread as cziRead


class Analysis(object):
    pathFile: str
    imageData: ImageData
    frames: list[Frame]

    def __init__(self, pathFile: str):
        self.pathFile = pathFile
        self.imageData = ImageData(
            cziRead(pathFile), CziFile(pathFile).metadata(False))
        self.frames = list(map(self.__genereteFrame, range(self.imageData.sizes.z-1)))

    def __genereteFrame(self, sliceId):
        return Frame(sliceId, self.imageData.getVoxelFrame(sliceId))
