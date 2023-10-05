
import os
import cv2
from warp import Curve, Distort, Stretch
from geometry import Rotate, Perspective, Shrink, TranslateX, TranslateY
from pattern import VGrid, HGrid, Grid, RectGrid, EllipseGrid
from noise import GaussianNoise, ShotNoise, ImpulseNoise, SpeckleNoise
from blur import GaussianBlur, DefocusBlur, MotionBlur, GlassBlur, ZoomBlur
from camera import Contrast, Brightness, JpegCompression, Pixelate
from weather import Fog, Snow, Frost, Rain, Shadow
from process import Posterize, Solarize, Invert, Equalize, AutoContrast, Sharpness, Color

from PIL import Image
import PIL.ImageOps
import numpy as np
import argparse
from dataset import clusterpixels
import matplotlib.pyplot as plt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', default="images/delivery.png", help='Load image file')
    parser.add_argument('--results', default="results", help='Load image file')
    parser.add_argument('--gray', action='store_true', help='Convert to grayscale 1st')
    opt = parser.parse_args()
    os.makedirs(opt.results, exist_ok=True)

    img = Image.open(opt.image)
    img = img.resize( (100,32) )
    mask_ = Image.fromarray(clusterpixels(img.convert("L"), 2))
    ops = [Curve(), Rotate(), Perspective(), Distort(), Stretch(), Shrink(), VGrid(), HGrid(), Grid(), RectGrid(), EllipseGrid()]
    ops.extend([GaussianNoise(), ShotNoise(), ImpulseNoise(), SpeckleNoise()])
    ops.extend([GaussianBlur(), DefocusBlur(), MotionBlur(), GlassBlur(), ZoomBlur()])
    ops.extend([Contrast(), Brightness(), JpegCompression(), Pixelate()])
    ops.extend([Fog(), Snow(), Frost(), Rain(), Shadow()])
    ops.extend([Posterize(), Solarize(), Invert(), Equalize(), AutoContrast(), Sharpness(), Color()])
    # for op in ops:
    #     for mag in range(3):
    #         mask = mask_.copy()
    #         filename = type(op).__name__ + "-" + str(mag) + ".png"
    #         filename_ = type(op).__name__ + "-" + str(mag) + "_mask.jpg"
    #         out_img, mask = op(img, mask, mag=mag)
    #         if opt.gray:
    #             out_img = PIL.ImageOps.grayscale(out_img)
    #         out_img.save(os.path.join(opt.results, filename))
    #         # mask.save(os.path.join(opt.results, filename_))
    #         plt.imshow(np.array(mask))
    #         plt.savefig(os.path.join(opt.results, filename_))
    #         plt.close()
    mask = mask_.copy()
    for aug in [Perspective(), Perspective()]:
        print(aug)
        op = aug
        mag = np.random.randint(0, 3)
        if type(op).__name__ == "Rain"  or type(op).__name__ == "Grid":
            img, mask = op(img.copy(), mask.copy(), mag=mag)
        else:
            img, mask = op(img, mask, mag=mag)
    filename = type(aug).__name__ + ".png"
    filename_ = type(aug).__name__ + "_mask.png"
    img.save(os.path.join(opt.results, filename))
    plt.imshow(np.array(mask))
    plt.savefig(os.path.join(opt.results, filename_))
    plt.close()


