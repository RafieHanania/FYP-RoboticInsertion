import math

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def wrap_to_pi(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi

class EMA:
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.y = None

    def update(self, x):
        if self.y is None:
            self.y = x
        else:
            self.y = self.alpha * x + (1 - self.alpha) * self.y
        return self.y
