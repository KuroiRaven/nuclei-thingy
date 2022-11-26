from ..services.utils import myfmad
from numpy import nanmedian, ndarray
from nucleus import Nucleus


class Frame(object):
    frameId: int
    voxel: ndarray
    nuclei: list[Nucleus]

    def __init__(self, frameId: int, voxel: ndarray):
        self.frameId = frameId
        self.voxel = voxel

    def getNormalizedVoxel(self) -> ndarray:
        return map(self.__normalizeSlice, self.voxel)

    def __normalizeSlice(self, voxelSlice):
        inter = voxelSlice - nanmedian(voxelSlice)
        inter /= myfmad(inter)
        return inter
