import logging
import pickle
import warnings

import numpy as np
from numpy import deprecate
from sklearn.decomposition import PCA
from joblib import dump, load


class EigenGrasp(object):
    
    # TODO: Is 2-dim enough?

    def __init__(self, original_dim, reduced_dim):
        self._N = 0
        self._D = original_dim
        self._M = reduced_dim
        self._pca = PCA(n_components=self._M)
        # Store the transformed joint angles for all principal components. This
        # is used to compute the min/max values of the transformed
        # representation.
        self._transformed_joint_angles = None

    @property
    def trained(self):
        """
        @:return True if the action space has been computed.
        """
        return hasattr(self._pca, 'components_')

    def dump_to_file(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self._pca, f)

    def load_from_file(self, file_path):
        with open(file_path, 'rb') as f:
            self._pca = pickle.load(f)
            assert self._pca.n_components == self._M, "PCA should be aligned with M"
        return self

    def fit_joint_values(self, joint_values):
        """
        Fit the principal components of the given joint values to compute the
        eigengrasp.

        Compute and store *all* D principal components here. The
        dimensionality of the lower-dimensional subspace is determined at grasp
        computation time.

        @:param joint_values A numpy array (N by D) with N datapoints of D
        dimensions.

        @:return True if the grasps were properly fit.

        """

        assert isinstance(joint_values, np.ndarray), 'Must have np.array().'

        if joint_values.size <= 0:
            return False

        self._N = joint_values.shape[0]
        self._D = joint_values.shape[1]
        self._transformed_joint_angles = self._pca.fit_transform(joint_values)

        print('Learned eigengrasp (from {}-dims) with {} points.'.
                     format(self._D, self._N))
        print(' Explained variance ratio: {}\b... ]'.format(
            self._pca.explained_variance_ratio_[0:4]))

        return True

    def reduce_original_dim(self, joint_values):
        """
        Reduce the original dimensionality of the joint values to the reduced
        dimensionality of the eigengrasp.

        @:param joint_values A numpy array (N by D) with N datapoints of D
        dimensions.

        @:return A numpy array (N by M) with N datapoints of M dimensions.

        """
        assert isinstance(joint_values, np.ndarray), 'Must have np.array().'

        if joint_values.size <= 0:
            return None

        return self._pca.transform(joint_values)

    def get_eigen_values_and_ratio(self):
        """
        Get the eigen values of the eigengrasp.
        """
        accumulate_ratio = [self._pca.explained_variance_ratio_[0]]
        for i in range(1, self._M):
            accumulate_ratio.append(accumulate_ratio[i-1] + self._pca.explained_variance_ratio_[i])
        return self._pca.explained_variance_, self._pca.explained_variance_ratio_, accumulate_ratio

    def compute_grasp(self, alphas):
        """
        Reconstruct a grasp given a combination of (low-dimensional) synergy
        coefficients to get a (full-dimensional) grasp configuration.

        The coefficients are a weighting vector of the various (ordered)
        principal grasp components.

        @:return mean + sum_i alpha_i * coeff_i. If the synergy is not already
        computed this returns None.

        """

        if not hasattr(self._pca, 'components_'):
            warnings.warn('No grasp synergies, did you call fit_joint_*?')
            return None
        #print('components:', self._pca.components_.shape)
        
        ret = self._pca.inverse_transform(alphas)

        return ret

    def joint_deg_range(self, component_num):
        """
        Compute the range of values for the i'th component of the joints,
        using the transformed original values used to train the PCA.

        If there are no joints or the component number is invalid, returns
        (0, 0).

        @:return (min, max) for transformed values
        """

        transformed_min, transformed_max = 0.0, 0.0

        # If there are no synergies, D = 0 so this does not get called.
        if 0 <= component_num < self._D:
            transformed_min = \
                np.min(self._transformed_joint_angles, axis=0)[component_num]
            transformed_max = \
                np.max(self._transformed_joint_angles, axis=0)[component_num]
        return transformed_min, transformed_max
