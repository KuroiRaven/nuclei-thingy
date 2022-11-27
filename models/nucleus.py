class Nucleus(object):
   #nucId: int
   #posX: int
   #poxY: int
   #posZ: int
   #spots: list

    def __init__(self, nucId, posX, posY, posZ):
        self.nucId = nucId
        self.posX = posX
        self.posY = posY
        self.posZ = posZ

    def getSpotAmount(self):
        return len(self.spots)
