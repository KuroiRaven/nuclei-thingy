from .sizes import ValuesAxis


class Spot(object):
    spotId: int
    pos: ValuesAxis
    intensity: float

    def __init__(self, spotId, posX, posY, posZ, intensity):
        self.spotId = spotId
        self.pos = ValuesAxis(posX, posY, posZ)
        self.intensity = intensity
