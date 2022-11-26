def getDistanceMetaData(metaDatas: dict, axis: str) -> float:
    distances = metaDatas["ImageDocument"]["Metadata"]["Scaling"]["Items"]["Distance"]
    lookedForDistance = next(
        filter(lambda distance: distance['Id'] == axis, distances), None)
    return lookedForDistance["Value"]
