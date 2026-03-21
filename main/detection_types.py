from dataclasses import dataclass

@dataclass
class Detection:
    t: float
    u: float
    v: float
    w: float
    h: float
    theta: float  # radians, canonicalised to [-pi/4, pi/4]
    conf: float