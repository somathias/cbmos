import numpy as _np
import numpy.random as _npr
import heapq as _hq
import logging as _logging

import time

from .. import cell as _cl

from ._eventqueue import EventQueue

_NU = 1


class CBModel:
    """
    Parameters
    -----------
        force: `f(ndarray(dtype=float), **kwargs)` -> float
            forces to be applied between cells
        solver: `f(fun, t_span, y0)` -> scipy.intergrade._ivp.ivp.OdeResult
            ODE solver, e.g. solve_ivp from scipy.integrate
        dimension: int
            dimension of the system, usually 2D or 3D
        separation: float
            distance between parent cell and child cell after separation
        hpc_backend: module
            module implementing Numpy's API (e.g. Cupy, Dask). Default is
            Numpy itself.

    """
    def __init__(self, force, solver, dimension=3, separation=0.3, hpc_backend=_np):
        self.force = force
        self.solver = solver
        self.dim = dimension
        self.separation = separation
        self.hpc_backend = hpc_backend

    def simulate(
            self,
            cell_list,
            t_data,
            force_args,
            solver_args,
            seed=None,
            raw_t=True,
            max_execution_time=None,
            min_event_resolution=0.,
            ):
        """
        Run the simulation with the given arguments and return the position
        of the cells at each time steps in `t_data`

        Parameters
        ----------
        cell_list: [Cell]
            Initial cell layout
        t_data: [float]
            times at which the history should be recorded, if `raw_t` is set to `True`,
            only the start and end time are taken into account and the rest is ignored.
            The history is then recorded following the solver's output.
        force_args: dict
            arguments to pass to the force function
        solver_args: dict
            arguments to pass to the solver
        seed: int
            seed for the random number generator
        raw_t: bool
            whether or not to use the solver's raw output. In that case, `t_data`
            is ignored and the raw times are returned along the history
        max_execution_time: float
            Maximum execution time in seconds that the simulation should use.
            Since the elapsed time is only checked in between cell events, this
            only represents an approximate target. The exact duration is saved
            in self.last_exec_time
        min_event_resolution: float
            Minimum event resolution interval: events occurring within
            `min_event_resolution` of the current time will be resolved
            immediately.

        Returns
        -------
        (t_data, history)

        Note
        ----
        - Cell ordering in the output can vary between timepoints.
        - Cell indices need to be unique for the whole duration of the simulation.
        - If `raw_t` is false, t_data is returned as is, with the history. If
          `raw_t` is true, aggregated t_data from the solver is returned.

        """

        exec_time_start = time.time()

        _npr.seed(seed)

        for cell in cell_list:
                assert len(cell.position) == self.dim

        _logging.debug("Starting new simulation")

        t = t_data[0]
        t_end = t_data[-1]
        self.cell_list = [
                _cl.Cell(
                    cell.ID, cell.position, cell.birthtime,
                    cell.proliferating, cell.generate_division_time,
                    cell.division_time, cell.parent_ID)
                for cell in cell_list]

        self.next_cell_index = max(self.cell_list, key=lambda cell: cell.ID).ID + 1
        self.history = []
        self.t_data = [t] if raw_t else t_data[:]
        self._save_data()

        # build event queue once, since independent of environment (for now)
        self._queue = EventQueue(
                [(cell.division_time, cell) for cell in self.cell_list],
                min_resolution=min_event_resolution,
                )

        while t < t_end:

            # check if max_execution_time has elapsed
            exec_time = time.time() - exec_time_start
            if max_execution_time is not None and exec_time >= max_execution_time:
                self.last_exec_time = exec_time
                return (self.t_data, self.history)

            # generate next event
            tau, cells = self._queue.pop()

            if tau > t:
                # calculate positions until the last t_data smaller or equal to min(tau, t_end)
                t_eval = [t] \
                    + [time for time in t_data if t < time <= min(tau, t_end)]
                y0 = _np.array([cell.position for cell in self.cell_list]).reshape(-1)
                # only calculate positions if there is t_data before tau
                if len(t_eval) > 1:
                    sol = self._calculate_positions(t_eval, y0, force_args, solver_args, raw_t=raw_t)

                    # save data for all t_data points passed
                    for y_t in sol.y[:, 1:].T:
                        self._save_data(y_t.reshape(-1, self.dim))
                    if raw_t:
                        self.t_data.extend(sol.t[1:])

                # check if max_execution_time has elapsed
                exec_time = time.time() - exec_time_start
                if max_execution_time is not None and exec_time >= max_execution_time:
                    self.last_exec_time = exec_time
                    return (self.t_data, self.history)

                # continue the simulation until tau if necessary
                if tau > t_eval[-1] and tau <=t_end:
                    y0 = sol.y[:, -1] if len(t_eval) > 1 else y0
                    sol = self._calculate_positions([t_eval[-1], tau], y0, force_args, solver_args, raw_t=raw_t)

                # update the positions for the current time point
                self._update_positions(sol.y[:, -1].reshape(-1, self.dim).tolist())

            # check if max_execution_time has elapsed
            exec_time = time.time() - exec_time_start
            if max_execution_time is not None and exec_time >= max_execution_time:
                self.last_exec_time = exec_time
                return (self.t_data, self.history)

            # apply event if tau <= t_end
            if tau <= t_end:
                for cell in cells:
                    self._apply_division(cell, tau)

            # update current time t to min(tau, t_end)
            t = min(tau, t_end)

        exec_time = time.time() - exec_time_start
        self.last_exec_time = exec_time
        return (self.t_data, self.history)

    def _save_data(self, positions=None):
        """
        Save the current positions of the cells to `self.history`. If
        `positions` is provided, uses theses positions instead of the cells'
        own positions.
        Note
        ----
        self.history has to be instantiated before the first call to _save_data
        as an empty list.
        """
        # copy correct positions to cell list that is stored
        if positions is not None:
            self.history.append([_cl.Cell(
                    cell.ID, pos, cell.birthtime, cell.proliferating,
                    cell.generate_division_time, cell.division_time,
                    cell.parent_ID)
                for cell, pos in zip(self.cell_list, positions)])
        else:
            self.history.append([_cl.Cell(
                    cell.ID, cell.position, cell.birthtime, cell.proliferating,
                    cell.generate_division_time, cell.division_time,
                    cell.parent_ID)
                for cell in self.cell_list])

    def _get_division_direction(self):

        if self.dim == 1:
            division_direction = _np.array([-1.0 + 2.0 * _npr.randint(2)])

        elif self.dim == 2:
            random_angle = 2.0 * _np.pi * _npr.rand()
            division_direction = _np.array([
                _np.cos(random_angle),
                _np.sin(random_angle)])

        elif self.dim == 3:
            u = _npr.rand()
            v = _npr.rand()
            random_azimuth_angle = 2 * _np.pi * u
            random_zenith_angle = _np.arccos(2 * v - 1)
            division_direction = _np.array([
                _np.cos(random_azimuth_angle) * _np.sin(random_zenith_angle),
                _np.sin(random_azimuth_angle) * _np.sin(random_zenith_angle),
                _np.cos(random_zenith_angle)])
        return division_direction

    def _apply_division(self, cell, tau):
        """
        Note
        ----
        The code assumes that all cell events are division events,
        """

        #check that the parent cell has set its proliferating flag to True
        assert cell.proliferating

        division_direction = self._get_division_direction()
        updated_position_parent = cell.position -\
            0.5 * self.separation * division_direction
        position_daughter = cell.position +\
            0.5 * self.separation * division_direction

        daughter_cell = _cl.Cell(
                self.next_cell_index, position_daughter, birthtime=tau,
                proliferating=True,
                division_time_generator=cell.generate_division_time,
                parent_ID=cell.ID)
        self.next_cell_index = self.next_cell_index + 1
        self.cell_list.append(daughter_cell)
        self._queue.push(daughter_cell.division_time, daughter_cell)

        cell.position = updated_position_parent
        cell.division_time = cell.generate_division_time(tau)
        self._queue.push(cell.division_time, cell)

        _logging.debug("Division event: t={}, direction={}".format(
            tau, division_direction))

    def _calculate_positions(self, t_eval, y0, force_args, solver_args, raw_t=True):
        return self.solver(self._ode_system(force_args),
                           (t_eval[0], t_eval[-1]),
                           y0,
                           t_eval=t_eval if not raw_t else None,
                           **solver_args)

    def _update_positions(self, y):
        """
        Note
        ----
        The ordering in cell_list and sol.y has to match.
        """

        for cell, pos in zip(self.cell_list, y):
            cell.position = _np.array(pos)

    def _ode_system(self, force_args):
        """ Generate ODE force function from cell-cell force function

        Parameters
        ----------
        force_args: {str: float}
            extra arguments for the force function

        Returns
        -------
        f: (t, y) -> dy/dt

        """
        def f(t, y):
            y_r = _np.expand_dims(
                    self.hpc_backend.asarray(y).reshape((-1, self.dim)),
                    axis=-1,
                    ) # shape (n, d, 1)
            cross_diff = y_r.transpose([2, 1, 0]) - y_r # shape (n, d, n)
            norm = _np.sqrt((cross_diff**2).sum(axis=1)) # shape (n, n)
            forces = _np.expand_dims(
                self.force(norm, **force_args)\
                    / (norm + _np.diag(self.hpc_backend.ones(y_r.shape[0]))),
                axis=1,
                ) # shape (n, 1, n)
            total_force = (forces * cross_diff).sum(axis=2) # shape (n, d)

            fty = (_NU*total_force).reshape(-1)

            if self.hpc_backend.__name__ == "cupy":
                return self.hpc_backend.asnumpy(fty)
            else:
                return _np.asarray(fty)

        return f

    def jacobian(self, y, force_args):
        """ Compute the jacobian of the given ode system.

        Parameters
        ----------
        y: np.ndarray(size=(n_cell*dim,))
            cell vector
        force_args: {str: float}
            extra arguments for the force function

        Returns
        -------
        np.ndarray(size=(n_cell*dim, n_cell*dim))
        """

        y_r = self.hpc_backend.asarray(_np.expand_dims(y.reshape((-1, self.dim)), axis=-1))
        n = y_r.shape[0]
        cross_diff = y_r - y_r.transpose([2, 1, 0]) # shape (n, d, n)
        norm = _np.sqrt((cross_diff**2).sum(axis=1))
        r_hat = _np.expand_dims(_np.moveaxis(cross_diff, 1, 2), axis=-1) # shape (n, n, d, 1)

        B = r_hat @ r_hat.transpose([0, 1, 3, 2]) # shape (n, n, d, d)

        with _np.errstate(divide='ignore', invalid='ignore'):
            # Ignore divide by 0 warnings
            # All NaNs are removed below

            # add normalization
            B = B / (norm*norm)[:, :, _np.newaxis, _np.newaxis]

            B = (
                    B*(self.force.derive()(norm, **force_args)-self.force(norm, **force_args)/norm)[:, :, _np.newaxis, _np.newaxis]
                    + (self.hpc_backend.identity(self.dim))[_np.newaxis, _np.newaxis, :, :]
                    * (self.force(norm, **force_args)/norm)[:, :, _np.newaxis, _np.newaxis]
                    )

            B[_np.isnan(B)] = 0

        # Step 2: compute the diagonal
        B[_np.array(range(n)), _np.array(range(n)), :, :] = - B.sum(axis=0)

        # Step 3: Build block matrix
        B_block =  B.reshape(n, n, self.dim, self.dim).swapaxes(1, 2).reshape(self.dim*n, -1)

        if self.hpc_backend.__name__ == "cupy":
            return self.hpc_backend.asnumpy(B_block)
        else:
            return _np.asarray(B_block)
