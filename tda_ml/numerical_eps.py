"""Named numerical floors (document in paper Methods / supplement)."""

# Denominators in metric tensor, probability weighting, aspect ratio.
NUMERICAL_EPS = 1e-8

# Ridge on local 2x2 covariance before ``eigh``.
PCA_RIDGE_EPS = 1e-6

# Floor before ``sqrt`` on PCA eigenvalues (non-negative spectrum).
EIGENVALUE_FLOOR = 1e-6

# Lower clamp on inlier probability in Mahalanobis distance weighting.
INLIER_PROB_MIN = 1e-4
