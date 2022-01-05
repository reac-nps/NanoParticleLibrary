import numpy as np


class CuttingPlane:
    """Class that represents a cutting plane via an anchor point and a normal."""

    def __init__(self, anchor, normal):
        self.anchor = anchor
        self.normal = normal

    def split_atom_indices(self, atoms):
        """Split the given atoms into the positive and negative subspace of the plane.

        Return numpy arrays of type boolean that are True if the atom with the corresponding
        index lays within the respective subspace.

        Parameters:
            atoms : Atoms
        """
        dot_product = np.dot((atoms.positions - self.anchor), self.normal)
        indices_in_positive_subspace = dot_product > 0
        indices_in_negative_subspace = dot_product < 0
        return indices_in_positive_subspace, indices_in_negative_subspace


class SphericalCuttingPlaneGenerator:
    """Class that generates cutting planes with a spherical anchor point distribution.

    The normal can be either parallel to the anchor point or point in a random
    direction. The former results in cuts that mainly affect the surface whereas
    the latter results in completely random cuts.
    """

    def __init__(self, max_radius, min_radius=0.0, center=0.0, normal_dir='parallel'):
        self.center = center
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.normal_dir = normal_dir

    def set_center(self, center):
        """Center around the anchor points will be distributed.

        Parameters:
            center : np.array of shape (3)
        """
        self.center = center

    def set_max_radius(self, max_radius):
        """Radius of the distribution where anchor point can be generated.

        Parameters:
            max_radius : float
        """
        self.max_radius = max_radius

    def generate_new_cutting_plane(self):
        """Generate a cutting plane by randomly choosing a normal and an anchor point."""
        normal = 1 - 2*np.random.random(3)
        normal /= np.linalg.norm(normal)

        if self.normal_dir == 'parallel':
            anchor_dir = normal
        else:
            anchor_dir = 1 - 2 * np.random.random(3)
            anchor_dir /= np.linalg.norm(anchor_dir)
        anchor = anchor_dir * (self.min_radius + np.random.random() * (self.max_radius - self.min_radius))
        anchor = anchor + self.center

        return CuttingPlane(anchor, normal)
