import numpy as np
import matplotlib.pyplot as plt

INTERIOR_ATOMS = 15
SPRING_CONSTANT = 1.0
LATTICE_CONSTANT = 1.0
INITIAL_AMPLITUDE = 0.05


def spring_force_1d(
    displacements_flat,
    spring_constant=SPRING_CONSTANT,
    interior_atoms=INTERIOR_ATOMS,
):
    """
    Return a vectorised force function for the 2-D spring lattice.

    The full grid has shape (N+2, N+2).
    Boundary rows/columns are pinned to zero displacement.
    Interior displacements are indexed [1:-1, 1:-1].

    For a scalar displacement field u (shape (N+2, N+2)):

        F[i,j] = k * (u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1] - 4*u[i,j])

    This is just the 2-D discrete Laplacian, which is computed
    with NumPy roll/shift operations.
    """
    N = interior_atoms
    disp = displacements_flat.reshape((N + 2, N + 2))
    force = np.zeros_like(disp)

    interior = disp[1:-1, 1:-1]
    force[1:-1, 1:-1] = spring_constant * (
        disp[0:-2, 1:-1]
        + disp[2:, 1:-1]
        + disp[1:-1, 0:-2]
        + disp[1:-1, 2:]
        - 4 * interior
    )

    return force.ravel()


def spring_force_2d(
    uv_flat,
    spring_constant=SPRING_CONSTANT,
    interior_atoms=INTERIOR_ATOMS,
):
    sz = (interior_atoms + 2) ** 2
    Fx = spring_force_1d(uv_flat[:sz], spring_constant, interior_atoms)
    Fy = spring_force_1d(uv_flat[sz:], spring_constant, interior_atoms)
    return np.concatenate([Fx, Fy])


def velocity_verlet_2d(
    u0_flat,
    force_fn,
    dt=2e-3,
    max_steps=2_000_000,
):
    """
    Velocity-Verlet integrator for a many-body system.
    """
    q = u0_flat.copy().astype(float)
    p = np.zeros_like(q)
    t = 0.0

    qs, ps, ts = [q.copy()], [p.copy()], [t]
    p_proj_prev = 0.0
    zero_crossings = 0

    # Unit vector along initial perturbation (for sign-change projection)
    u0_norm = np.linalg.norm(u0_flat)
    n_hat = u0_flat / u0_norm if u0_norm > 0 else u0_flat

    for _ in range(max_steps):
        F = force_fn(q)
        p_half = p + 0.5 * dt * F
        q = q + dt * p_half
        p = p_half + 0.5 * dt * force_fn(q)
        t += dt

        qs.append(q.copy())
        ps.append(p.copy())
        ts.append(t)

        # Scalar sign-change criterion: project p onto initial mode
        p_proj = np.dot(p, n_hat)
        if len(qs) > 2 and p_proj_prev * p_proj < 0.0:
            zero_crossings += 1
            if zero_crossings == 2:
                break
        p_proj_prev = p_proj
    else:
        return None

    return np.array(qs), np.array(ps), np.array(ts)


def make_initial_displacement(
    nx, ny, amplitude=INITIAL_AMPLITUDE, interior_atoms=INTERIOR_ATOMS
):
    """
    Build a sub-harmonic sinusoidal perturbation on the interior grid.

    For wavevector k = (kx, ky) = (π·nx/(N+1), π·ny/(N+1)):

        φ(i,j) = A · sin(kx·i) · sin(ky·j)     (scalar envelope)

    """
    i_idx = np.arange(interior_atoms + 2)
    j_idx = np.arange(interior_atoms + 2)
    ii, jj = np.meshgrid(i_idx, j_idx, indexing="ij")

    kx = np.pi * nx / (interior_atoms + 1)
    ky = np.pi * ny / (interior_atoms + 1)
    theta = np.arctan2(ky, kx)

    phi = amplitude * np.sin(kx * ii) * np.sin(ky * jj)

    # LA displacement
    ux = phi * np.cos(theta)
    uy = phi * np.sin(theta)

    # # TA displacement
    # ux = -phi * np.sin(theta)
    # uy = phi *   np.cos(theta)

    # fixed boundary conditions
    for u in (ux, uy):
        u[0, :] = u[-1, :] = u[:, 0] = u[:, -1] = 0.0

    return np.concatenate([ux.ravel(), uy.ravel()]), kx, ky


def plot_lattice_snapshot(
    uv_flat,
    interior_atoms=INTERIOR_ATOMS,
    lattice_constant=LATTICE_CONSTANT,
    ax=None,
    scale=5.0,
    title=None,
    show_bonds=True,
):
    """
    Plot a snapshot of the 2D lattice given a flat displacement vector.

    Parameters
    ----------
    uv_flat       : 1D array of length 2*(N+2)**2
                    Concatenated [ux.ravel(), uy.ravel()] from the integrator.
    interior_atoms: int, N (grid is (N+2)×(N+2))
    lattice_constant: float, equilibrium spacing a
    ax            : matplotlib Axes, or None to create a new figure
    scale         : float, displacement magnification for visibility
    title         : str, optional axes title
    show_bonds    : bool, draw equilibrium bonds between nearest neighbours

    Returns
    -------
    fig, ax
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
    else:
        fig = ax.get_figure()

    N = interior_atoms
    Ng = N + 2
    sz = Ng * Ng

    # Unpack displacements
    ux = uv_flat[:sz].reshape(Ng, Ng)
    uy = uv_flat[sz:].reshape(Ng, Ng)

    # Equilibrium positions
    idx = np.arange(Ng) * lattice_constant
    x0, y0 = np.meshgrid(idx, idx, indexing="ij")

    # Displaced positions
    x = x0 + scale * ux
    y = y0 + scale * uy

    # Displacement magnitude for colouring (boundary atoms always 0)
    mag = np.sqrt(ux**2 + uy**2)

    # Bonds
    if show_bonds:
        for i in range(Ng):
            for j in range(Ng):
                if i + 1 < Ng:  # vertical bond
                    ax.plot(
                        [x[i, j], x[i + 1, j]],
                        [y[i, j], y[i + 1, j]],
                        color="#c8c8c8",
                        lw=0.6,
                        zorder=1,
                    )
                if j + 1 < Ng:  # horizontal bond
                    ax.plot(
                        [x[i, j], x[i, j + 1]],
                        [y[i, j], y[i, j + 1]],
                        color="#c8c8c8",
                        lw=0.6,
                        zorder=1,
                    )

    # Atoms
    is_boundary = np.zeros((Ng, Ng), dtype=bool)
    is_boundary[0, :] = is_boundary[-1, :] = True
    is_boundary[:, 0] = is_boundary[:, -1] = True

    vmax = mag[1:-1, 1:-1].max()
    vmax = vmax if vmax > 0 else 1.0

    ax.scatter(
        x[~is_boundary],
        y[~is_boundary],
        c=mag[~is_boundary],
        cmap="plasma",
        vmin=0,
        vmax=vmax,
        s=100,
        zorder=3,
        linewidths=0.4,
        edgecolors="k",
    )

    ax.scatter(
        x[is_boundary],
        y[is_boundary],
        color="#888888",
        s=75,
        zorder=3,
        linewidths=0.4,
        edgecolors="#444444",
        marker="s",
    )

    # Displacement
    ax.quiver(
        x0[~is_boundary],
        y0[~is_boundary],
        ux[~is_boundary],
        uy[~is_boundary],
        scale_units="xy",
        scale=1.0 / scale,
        width=0.003,
        color="steelblue",
        alpha=0.5,
        zorder=2,
    )

    ax.set(
        aspect="equal",
        xlim=[-0.5, (Ng - 1) * lattice_constant + 0.5],
        ylim=[-0.5, (Ng - 1) * lattice_constant + 0.5],
        xticks=[],
        yticks=[],
    )

    return fig, ax


def visualize_orbit(qs, ts, every=50, interior_atoms=INTERIOR_ATOMS, **kwargs):
    """
    Produce a grid of snapshots sampled uniformly across one orbit.
    """
    frames = qs[::every]
    times = ts[::every]
    n = len(frames)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4 * ncols, 4 * nrows), constrained_layout=True
    )
    axes = np.array(axes).ravel()

    for k, (uv, t) in enumerate(zip(frames, times)):
        plot_lattice_snapshot(
            uv,
            interior_atoms=interior_atoms,
            ax=axes[k],
            title=f"t = {t:.3f}",
            **kwargs,
        )

    for ax in axes[n:]:
        ax.set_visible(False)

    return fig
