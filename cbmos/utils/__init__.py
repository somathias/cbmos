import numpy as _np
import numpy.random as _npr
import matplotlib.pyplot as plt


from .. import cell as _cl

def generate_cartesian_coordinates(n_x, n_y, scaling=1.0):
    """
    Generate cartesian coordinates which can be used to set up a cell
    population.
    """
    return [(scaling*i_x, scaling*i_y)
            for i_x in range(n_x) for i_y in range(n_y)
            ]


def generate_honeycomb_coordinates(n_x, n_y, scaling=1.0):
    """
    Generate coordinates from a honeycomb mesh which can be used to set up a
    cell population.
    """
    return [((2 * i_x + (i_y % 2)) * 0.5 * scaling,
             _np.sqrt(3) * i_y * 0.5 * scaling)
            for i_x in range(n_x) for i_y in range(n_y)
            ]


def setup_locally_compressed_monolayer(n_x, n_y, scaling=1.0, separation=0.3):
    """
    Set up a locally compressed monolayer where the middle cell has just
    divided.

    Returns: list of cells

    """

    coords = generate_honeycomb_coordinates(n_x, n_y, scaling=scaling)
    sheet = [
            _cl.Cell(i, [x, y], 0.0, proliferating=False)
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
    daughter_cell = _cl.Cell(next_cell_index, position_daughter,
                             birthtime=0.0, proliferating=False)
    sheet.append(daughter_cell)
    return sheet



def plot_2d_population(cell_list, color='blue'):
    """Plot a two dimensional population provided as a list of Cell objects."""

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for cell in cell_list:
        ax.add_patch(plt.Circle(cell.position, 0.5, color=color, alpha=0.4))
        plt.plot(cell.position[0], cell.position[1], '.', color=color)
    ax.set_aspect('equal')
    plt.show()
