import numpy as np
import cv2
import imgaug.augmenters as iaa

def main():
    n = 100
    image = cv2.imread('test.jpg')
    images = np.stack([image]*n)

    print(images.shape)

    seq = iaa.Sequential([
        iaa.Crop(percent=(0, 0.1)), # random crops
        iaa.Sometimes(
            0.5,
            iaa.GaussianBlur(sigma=(0, 0.3))
        ),
        iaa.ChannelShuffle(0.5),
        iaa.Dropout(p=(0, 0.2)),
        iaa.LinearContrast((0.75, 1.5)),
        iaa.CoarseDropout((0.02, 0.1), size_percent=0.2),
        # iaa.Dropout2d(p=0.5),
        iaa.Invert(0.1),
        iaa.Solarize(0.1, threshold=(32, 128)),
        # iaa.Cartoon(),
        # iaa.BlendAlphaRegularGrid(nb_rows=(10, 20), nb_cols=(10, 20), foreground=iaa.Multiply(0.0)),
        # iaa.BlendAlphaCheckerboard(nb_rows=2, nb_cols=(1, 4), foreground=iaa.AddToHue((-100, 100))),
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
        iaa.Multiply((0.8, 1.2), per_channel=0.2),
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        )
    ], random_order=True) # apply augmenters in random order

    images_aug = seq(images=images)
    for idx, img in enumerate(images_aug):
        img_name = str(idx)+'.jpg'
        cv2.imwrite(img_name, img)


if __name__ == '__main__':
    main()
