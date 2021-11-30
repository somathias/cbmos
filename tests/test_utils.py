import numpy as np
import pytest

import cbmos.utils as ut


def test_hcp_grid():

    n_x = 3
    n_y = 3
    n_z = 3

    scaling = 0.8

    coords = ut.generate_hcp_coordinates(n_x, n_y, n_z, scaling=0.8)

    assert len(coords) == n_x*n_y*n_z

    coords_first = np.array(coords[0])
    coords_second = np.array(coords[1])
    coords_last = np.array(coords[-1])

    # check that first node is in origin
    assert np.all(coords_first == 0.0)

    # check that first two have distance scaling
    assert np.linalg.norm(coords_second-coords_first) == pytest.approx(scaling)

    # check the coordinates of the last node
    assert coords_last[0] == pytest.approx(scaling*2)
    assert coords_last[1] == pytest.approx(scaling*1.7320, 1e-4)
    assert coords_last[2] == pytest.approx(scaling*1.6329, 1e-4)