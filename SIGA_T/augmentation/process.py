
from PIL import Image
import PIL.ImageOps, PIL.ImageEnhance
import numpy as np

class Posterize:
    def __init__(self):
        pass

    def __call__(self, img, mask, mag=-1, prob=1.):
        if np.random.uniform(0,1) > prob:
            return img, mask

        c = [1, 3, 6]
        if mag<0 or mag>=len(c):
            index = np.random.randint(0, len(c))
        else:
            index = mag
        c = c[index]
        bit = np.random.randint(c, c+2)
        img = PIL.ImageOps.posterize(img, bit)

        return img, mask


class Solarize:
    def __init__(self):
        pass

    def __call__(self, img, mask, mag=-1, prob=1.):
        if np.random.uniform(0,1) > prob:
            return img, mask

        c = [64, 128, 192]
        if mag<0 or mag>=len(c):
            index = np.random.randint(0, len(c))
        else:
            index = mag
        c = c[index]
        thresh = np.random.randint(c, c+64)
        img = PIL.ImageOps.solarize(img, thresh)

        return img, mask

class Invert:
    def __init__(self):
        pass

    def __call__(self, img, mask, mag=-1, prob=1.):
        if np.random.uniform(0,1) > prob:
            return img, mask

        img = PIL.ImageOps.invert(img)

        return img, mask

    
class Equalize:
    def __init__(self):
        pass

    def __call__(self, img, mask, mag=-1, prob=1.):
        if np.random.uniform(0,1) > prob:
            return img, mask

        img = PIL.ImageOps.equalize(img)

        return img, mask


class AutoContrast:
    def __init__(self):
        pass

    def __call__(self, img, mask, mag=-1, prob=1.):
        if np.random.uniform(0,1) > prob:
            return img, mask

        img = PIL.ImageOps.autocontrast(img)

        return img, mask


class Sharpness:
    def __init__(self):
        pass

    def __call__(self, img, mask, mag=-1, prob=1.):
        if np.random.uniform(0,1) > prob:
            return img, mask

        c = [.1, .7, 1.3]
        if mag<0 or mag>=len(c):
            index = np.random.randint(0, len(c))
        else:
            index = mag
        c = c[index]
        magnitude = np.random.uniform(c, c+.6)
        img = PIL.ImageEnhance.Sharpness(img).enhance(magnitude)

        return img, mask


class Color:
    def __init__(self):
        pass

    def __call__(self, img, mask, mag=-1, prob=1.):
        if np.random.uniform(0,1) > prob:
            return img, mask

        c = [.1, .7, 1.3]
        if mag<0 or mag>=len(c):
            index = np.random.randint(0, len(c))
        else:
            index = mag
        c = c[index]
        magnitude = np.random.uniform(c, c+.6)
        img = PIL.ImageEnhance.Color(img).enhance(magnitude)

        return img, mask


