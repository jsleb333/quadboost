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
            [scale_x*np.sin(rotation+shear_x),  scale_y*np.cos(rotation+shear_y), t_y]
        ])

        self.center = np.array(center).reshape(2,1)
        center_translation = self.affine_matrix[:2,:2].dot(self.center)
        self.affine_matrix[:2,2:3] += self.center - center_translation

    def __repr__(self):
        return repr(self.affine_matrix)

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
        if len(input_matrix.shape) == 3:
            transformed = np.array(
                [affine_transform(ch, self.affine_matrix, **kwargs) for ch in input_matrix])
        else:
            transformed = affine_transform(input_matrix, self.affine_matrix, **kwargs)
        return transformed
