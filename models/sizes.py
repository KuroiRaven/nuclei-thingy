from typing import TypeVar, Generic, List

T = TypeVar('T')


class ValuesAxis(Generic[T]):
    x: T
    y: T
    z: T

    def __init__(self, x: T, y: T, z: T) -> None:
        self.x = x
        self.y = y
        self.z = z
