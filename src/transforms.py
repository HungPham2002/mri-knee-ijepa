import torchio as tio

def make_transforms(training=True, **kwargs):
    if training:
        spatial_augment = [
            tio.RandomAffine(
                scales=(0.9, 1.1),
                degrees=15,
                translation=5,
                p=0.5,
            ),
            tio.RandomFlip(axes=(0,), flip_probability=0.5),
        ]

        intensity_augment = {
            tio.RandomNoise(std=(0, 0.1)): 0.2,
            tio.RandomBiasField(coefficients=0.5): 0.2,
            tio.RandomBlur(std=(0, 1.5)): 0.2,
            tio.RandomMotion(num_transforms=2): 0.2,
            tio.RandomGamma(log_gamma=(-0.3, 0.3)): 0.2,
        }

        # Torchio transforms list
        return tio.Compose([
            tio.Compose(spatial_augment, p=0.8),
            tio.OneOf(intensity_augment, p=0.8),
            tio.RescaleIntensity(out_min_max=(-1, 1)),
        ])
    else:
        return tio.Compose([
            tio.RescaleIntensity(out_min_max=(-1, 1)),
        ])
