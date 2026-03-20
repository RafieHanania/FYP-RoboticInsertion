from dataclasses import dataclass

@dataclass
class Detection:
    t: float
    u: float
    v: float
    w: float
    h: float
    theta: float # radians -pi/2 to pi/2
    conf: float