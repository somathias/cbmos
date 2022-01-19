import numpy as _np
import numpy.random as _npr
import matplotlib.pyplot as _plt


from .. import cell as _cl

def generate_cartesian_coordinates(n_x, n_y, scaling=1.0):
    """
    Generate cartesian coordinates which can be used to set up a cell
    population.

    Parameters
    ----------
    n_x: int
        number of columns
    n_y: int
        number of rows
    scaling: float
        distance between the cells, in cell diameters
    """
    return [(scaling*i_x, scaling*i_y)
            for i_x in range(n_x) for i_y in range(n_y)
            ]


def generate_honeycomb_coordinates(n_x, n_y, scaling=1.0):
    """
    Generate coordinates from a honeycomb mesh which can be used to set up a
    cell population.

    Parameters
    ----------
    n_x: int
        number of columns
    n_y: int
        number of rows
    scaling: float
        distance between the cells, in cell diameters
    """
    return [((2 * i_x + (i_y % 2)) * 0.5 * scaling,
             _np.sqrt(3) * i_y * 0.5 * scaling)
            for i_x in range(n_x) for i_y in range(n_y)
            ]


def generate_hcp_coordinates(n_x, n_y, n_z, scaling=1.0):
    """
    Generate coordinates from a HCP (hexagonal close-packed) grid which can be
    used to set up a cell population.

    Parameters
    ----------
    n_x: int
        number of columns
    n_y: int
        number of rows
    n_z: int
        number of layers
    scaling: float
        distance between the cells, in cell diameters
    """
    return [((2 * i_x + ((i_y + i_z) % 2)) * 0.5 * scaling,
             _np.sqrt(3) * (i_y + (i_z % 2) / 3.0) * 0.5 * scaling,
             _np.sqrt(6) * i_z / 3.0 * scaling)
            for i_x in range(n_x) for i_y in range(n_y) for i_z in range(n_z)
            ]


def setup_locally_compressed_monolayer(n_x, n_y, scaling=1.0, separation=0.3):
    """
    Set up a locally compressed monolayer where the middle cell has just
    divided.

    Parameters
    ----------
    n_x: int
        number of columns
    n_y: int
        number of rows
    scaling: float
        distance between neighboring cells, in cell diameters
    separation: float
        distance between daughter cells, in cell diameters

    Returns
    -------
        list of cells

    """

    coords = generate_honeycomb_coordinates(n_x, n_y, scaling=scaling)
    sheet = [
            _cl.Cell(i, [x, y])
            for i, (x, y) in enumerate(coords)
            ]

    # find middle index
    m = n_x * (n_y//2) + n_x//2
    coords = list(sheet[m].position)

    # get division direction
    random_angle = 2.0 * _np.pi * _npr.rand()
    division_direction = _np.array([_np.cos(random_angle),
                                    _np.sin(random_angle)])

    # update positions
    updated_position_parent = coords - 0.5 * separation * division_direction
    sheet[m].position = updated_position_parent

    position_daughter = coords + 0.5 * separation * division_direction

    # add daughter cell
    next_cell_index = len(sheet)
    daughter_cell = _cl.Cell(next_cell_index, position_daughter)
    sheet.append(daughter_cell)
    return sheet

def setup_locally_compressed_spheroid(n_x, n_y, n_z, scaling=1.0, separation=0.3):
    """
    Set up a locally compressed spheroid where the middle cell has just
    divided.

    Parameters
    ----------
    n_x: int
        number of columns
    n_y: int
        number of rows
    n_z: int
        number of layers
    scaling: float
        distance between neighboring cells, in cell diameters
    separation: float
        distance between daughter cells, in cell diameters

    Returns
    -------
        list of cells

    """

    coords = [((2 * i_x + ((i_y + i_z) % 2)) * 0.5 * scaling,
             _np.sqrt(3) * (i_y + (i_z % 2) / 3.0) * 0.5 * scaling,
             _np.sqrt(6) * i_z / 3.0 * scaling)
            for i_x in range(n_x) for i_y in range(n_y) for i_z in range(n_z)
            ]


    # make cell_list for the sheet
    sheet = [_cl.Cell(i, [x,y,z]) for i, (x, y, z) in enumerate(coords)]


    # find middle index, move cell there and add second daughter cells
    m = (n_x*n_y)*(n_z//2)+n_x*(n_y//2)+n_x//2
    coords = list(sheet[m].position)

    # get division direction
    u = _npr.rand()
    v = _npr.rand()
    random_azimuth_angle = 2 * _np.pi * u
    random_zenith_angle = _np.arccos(2 * v - 1)
    division_direction = _np.array([
                _np.cos(random_azimuth_angle) * _np.sin(random_zenith_angle),
                _np.sin(random_azimuth_angle) * _np.sin(random_zenith_angle),
                _np.cos(random_zenith_angle)])

    # update positions
    updated_position_parent = coords - 0.5 * separation * division_direction
    sheet[m].position = updated_position_parent

    position_daughter = coords + 0.5 * separation * division_direction

    # add daughter cell
    next_cell_index = len(sheet)
    daughter_cell = _cl.Cell(next_cell_index, position_daughter)
    sheet.append(daughter_cell)

    return sheet



def plot_2d_population(cell_list, color='blue'):
    """Plot a two dimensional population provided as a list of Cell objects."""

    fig = _plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for cell in cell_list:
        ax.add_patch(_plt.Circle(cell.position, 0.5, color=color, alpha=0.4))
        _plt.plot(cell.position[0], cell.position[1], '.', color=color)
    ax.set_aspect('equal')
    _plt.show()
