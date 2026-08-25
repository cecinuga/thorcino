import numpy as np

class RandomHorizontalFlip:
    def __init__(self, p:float=0.5):
        self.p = p

    def __call__(self, x):
        if np.random.random() < self.p:
            # Flip along width axis (last axis)
            return np.flip(x, axis=-1).copy()
        return x

class RandomCrop:
    def __init__(self, height:int=0, width:int=0, padding:int=0):
        self.padding:int = padding
        self.h:int = height
        self.w:int = width

    def __call__(self, data) -> np.ndarray:
        # 1. Pad image borders with zeros
        padded: np.ndarray = np.pad(data, self.padding, mode='constant')

        # 2. Select random top-left corner
        top = np.random.randint(0, 2*self.padding+1)
        left = np.random.randint(0, 2*self.padding+1)

        # 3. Slice back to original size
        cropped = padded[..., top:top+self.h, left:left+self.w]
        return cropped

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)

        return x
