import numpy as np
from scipy.ndimage import affine_transform


class AffineTransform:
    """
    Computes the affine transformation from affine parameters and applies it to a matrix to transform, using its indices as coordinates.
    """
    def __init__(self, rotation=0, scale=1, shear=0, translation=(0,0), center=(0,0)):
        """
        Computes the affine transformation matrix given the parameters.

        Args:
            rotation (float): Angle of rotation in radians.
            scale (float or tuple of floats): Scale factors. If only one factor is specified, scaling in both direction will be the same.
            shear (float or tuple of floats): Angles of shear in radians. If only one angle is specified, the shear is applied in the x axis only.
            translation (tuple of floats): Translation to apply after the rotation, shear and scaling.
            center (tuple of floats): Position in the image from which the transformation is applied.
        """
        self.rotation = rotation
        self.scale = (scale, scale) if isinstance(scale, (int, float)) else scale
        scale_x, scale_y = self.scale
        self.shear = (shear, 0)  if isinstance(shear, (int, float)) else shear
        shear_x, shear_y = self.shear
        self.translation = np.array(translation)
        t_x, t_y = self.translation

        self.affine_matrix = np.array([
            [scale_x*np.cos(rotation+shear_x), -scale_y*np.sin(rotation+shear_y), t_x],
            [scale_x*np.sin(rotation+shear_x),  scale_y*np.cos(rotation+shear_y), t_y],
            [                               0,                                 0,   1]
        ])

        self.center = np.array(center).reshape(2,1)
        center_translation = self.affine_matrix[:2,:2].dot(self.center)
        self.affine_matrix[:2,2:3] += self.center - center_translation

    def __repr__(self):
        return repr(self.affine_matrix)

    def __bool__(self):
        no_rotation = self.rotation == 0
        no_scale = self.scale[0] == 1 and self.scale[1] == 1
        no_shear = self.shear[0] == 0 and self.shear[1] == 0
        no_translation = self.translation[0] == 0 and self.translation[1] == 0
        if no_rotation and no_scale and no_shear and no_translation:
            return False
        else:
            return True

    def __call__(self, input_matrix, **kwargs):
        """
        Applies the affine transformation on the input matrix, using its indices as coordinates.

        Args:
            input_matrix (numpy array): Matrix to transform.
            kwargs: Keyword arguments of scipy.ndimage.affine_transform. Defaults are:
                offset=0.0
                output_shape=None
                output=None
                order=3
                mode='constant'
                cval=0.0
                prefilter=True
        """
        if not self: # Is identity transformation
            return input_matrix

        if len(input_matrix.shape) == 3:
            transformed = np.array(
                [affine_transform(ch, self.affine_matrix, **kwargs) for ch in input_matrix])
        else:
            transformed = affine_transform(input_matrix, self.affine_matrix, **kwargs)
        return transformed


def random_affine(rotation=0, scale_x=0, scale_y=0, shear_x=0, shear_y=0, translation_x=0, translation_y=0, center=(0,0), angle_unit='radians'):
    """
    Generates a random AffineTransform object given the parameters ranges.

    Args:
        rotation (float or tuple of floats): If a single number, a random angle will be picked between (-rotation, rotation) uniformly. Else, a random angle will be picked between the two specified numbers.

        scale_x (float or tuple of floats): Relative scaling factor. If a single number, a random scale factor will be picked between (1-scale_x, 1/(1-scale_x)) uniformly. Else, a random scale factor will be picked between the two specified numbers.

        scale_y (float or tuple of floats): Relative scaling factor. Same a scale_x but for y direction.

        shear_x (float or tuple of floats): Shear angle in x direction. If a single number, a random angle will be picked between (-shear_x, shear_x) uniformly. Else, a random angle will be picked between the two specified numbers.

        shear_y (float or tuple of floats): Same as shear_x but for y direction.

        translation_x (float or tuple of floats): Number of pixels translation in x direction. If a single number, a random number will be picked between (-translation_x, translation_x) uniformly. Else, a random number will be picked between the two specified numbers.
        translation_y (float or tuple of floats): Same as translation_x but for y direction.

        center (tuple of floats): Position to apply the AffineTransform. It is not picked at random.

        angle_unit ('radians' or 'degrees'): Specifies the angles of the rotation and shear passed to the function. If in degrees, the will be converted to radians.

    Returns: AffineTransform object.
    """
    if angle_unit == 'degrees':
        rotation = np.deg2rad(rotation)
        shear_x = np.deg2rad(shear_x)
        shear_y = np.deg2rad(shear_y)

    if isinstance(rotation, (int, float)): rotation = (-rotation, rotation)
    random_rot = np.random.uniform(*rotation)

    if isinstance(scale_x, (int, float)): scale_x = (1-scale_x, 1/(1-scale_x))
    if isinstance(scale_y, (int, float)): scale_y = (1-scale_y, 1/(1-scale_y))
    random_scale = (np.random.uniform(*scale_x), np.random.uniform(*scale_y))

    if isinstance(shear_x, (int, float)): shear_x = (-shear_x, shear_x)
    if isinstance(shear_y, (int, float)): shear_y = (-shear_y, shear_y)
    random_shear = (np.random.uniform(*shear_x), np.random.uniform(*shear_y))

    if isinstance(translation_x, (int, float)): translation_x = (-translation_x, translation_x)
    if isinstance(translation_y, (int, float)): translation_y = (-translation_y, translation_y)
    random_translation = (np.random.uniform(*translation_x), np.random.uniform(*translation_y))

    return AffineTransform(random_rot, random_scale, random_shear, random_translation, center)


class RandomAffine:
    """
    Samples random affine transformations given the parameters ranges.
    """
    def __init__(self, rotation=0, scale_x=0, scale_y=0, shear_x=0, shear_y=0, translation_x=0, translation_y=0, angle_unit='radians'):
        """
        Generates a random affine transform given the parameters ranges.

        Args:
            rotation (float or tuple of floats): If a single number, a random angle will be picked between (-rotation, rotation) uniformly. Else, a random angle will be picked between the two specified numbers.

            scale_x (float or tuple of floats): Relative scaling factor. If a single number, a random scale factor will be picked between (1-scale_x, 1/(1-scale_x)) uniformly. Else, a random scale factor will be picked between the two specified numbers.

            scale_y (float or tuple of floats): Relative scaling factor. Same a scale_x but for y direction.

            shear_x (float or tuple of floats): Shear angle in x direction. If a single number, a random angle will be picked between (-shear_x, shear_x) uniformly. Else, a random angle will be picked between the two specified numbers.

            shear_y (float or tuple of floats): Same as shear_x but for y direction.

            translation_x (float or tuple of floats): Number of pixels translation in x direction. If a single number, a random number will be picked between (-translation_x, translation_x) uniformly. Else, a random number will be picked between the two specified numbers.
            translation_y (float or tuple of floats): Same as translation_x but for y direction.

            center (tuple of floats): Position to apply the AffineTransform. It is not picked at random.

            angle_unit ('radians' or 'degrees'): Specifies the angles of the rotation and shear passed to the function. If in degrees, the will be converted to radians.
        """
        if angle_unit == 'degrees':
            rotation = np.deg2rad(rotation)
            shear_x = np.deg2rad(shear_x)
            shear_y = np.deg2rad(shear_y)

        self.rotation = (-rotation, rotation) if isinstance(rotation, (int, float)) else rotation

        self.scale_x = (1-scale_x, 1/(1-scale_x)) if isinstance(scale_x, (int, float)) else scale_x
        self.scale_y = (1-scale_y, 1/(1-scale_y)) if isinstance(scale_y, (int, float)) else scale_y

        self.shear_x = (-shear_x, shear_x) if isinstance(shear_x, (int, float)) else shear_x
        self.shear_y = (-shear_y, shear_y) if isinstance(shear_y, (int, float)) else shear_y

        self.translation_x = (-translation_x, translation_x) if isinstance(translation_x, (int, float)) else translation_x
        self.translation_y = (-translation_y, translation_y) if isinstance(translation_y, (int, float)) else translation_y

    def sample_transformation(self, center=(0,0)):
        random_rot = np.random.uniform(*self.rotation)
        random_scale = (np.random.uniform(*self.scale_x), np.random.uniform(*self.scale_y))
        random_shear = (np.random.uniform(*self.shear_x), np.random.uniform(*self.shear_y))
        random_translation = (np.random.uniform(*self.translation_x), np.random.uniform(*self.translation_y))

        return AffineTransform(random_rot, random_scale, random_shear, random_translation, center)


if __name__ == '__main__':
    print(random_affine(rotation=15, scale_x=.1, shear_x=10, scale_y=.1, shear_y=10,
                        center=(14,14), angle_unit='degrees'))
