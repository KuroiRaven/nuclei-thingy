from .sizes import ValuesAxis
from .spot import Spot
from numpy import ndarray


class Nucleus(object):
    nucId: int
    composition: ndarray
    pos: ValuesAxis[float]
    spots: list[Spot]

    def __init__(self, nucId, posX, posY, posZ):
        self.nucId = nucId
        self.pos = ValuesAxis(posX, posY, posZ)

    def getSpotAmount(self):
        return len(self.spots)
