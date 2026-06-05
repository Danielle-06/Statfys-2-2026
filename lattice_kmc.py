"""
Kinetic Monte Carlo for a 2D square lattice.

Species encoding
----------------
    vacancy =  0
    blue    = -1
    red     = +1

Chemical potential convention
-------------------------------------------------------------------
    High muB   →  blue atoms are energetically favoured
    High muR   →  red  atoms are energetically favoured

    Adding a blue atom:    dE -= muB   (lowers energy → favours insertion)
    Removing a blue atom:  dE += muB   (raises energy → penalises removal)

Modes
-----
    0  MODE_CANONICAL               — vacancy-mediated swap
    1  MODE_GRAND_CANONICAL         — bulk flip anywhere in interior
    2  MODE_SURFACE_GRAND_CANONICAL — vacancy swap + edge-band insertion
"""

import numpy as np
from numba import njit

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from matplotlib.patches import Circle, Rectangle
from matplotlib.collections import PatchCollection
from ipycanvas import MultiCanvas, hold_canvas

from ipywidgets import (
    Play,
    IntSlider,
    FloatSlider,
    Dropdown,
    HBox,
    VBox,
    jslink,
    Layout,
    Button,
)
from IPython.display import display

MODE_CANONICAL = 0
MODE_GRAND_CANONICAL = 1
MODE_SURFACE_GRAND_CANONICAL = 2

DIRECTIONS = np.array(
    [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)],
    dtype=np.int64,
)


def initialize_lattice(shape, probs=(0.1, 0.45, 0.45), seed=None):
    """
    Fill interior randomly, leaving a width-3 vacancy border.
    Returns (lattice, vacancy_coords) where vacancy_coords is an (N,2) int64
    array of all vacancy positions inside the interior.
    """
    rng = np.random.default_rng(seed)
    lat = np.zeros(shape, dtype=np.int64)
    interior = lat[3:-3, 3:-3]
    interior[:] = rng.choice(np.array([0, -1, 1]), size=interior.shape, p=probs)

    rows, cols = shape
    vac = []
    for i in range(3, rows - 3):
        for j in range(3, cols - 3):
            if lat[i, j] == 0:
                vac.append((i, j))

    vacancy_coords = (
        np.array(vac, dtype=np.int64) if vac else np.empty((0, 2), dtype=np.int64)
    )
    return lat, vacancy_coords


@njit
def local_energy_change_swap(lattice, i, j, ni, nj, J1, J2):
    """ΔE for swapping lattice[i,j] ↔ lattice[ni,nj]."""
    s1 = lattice[i, j]
    s2 = lattice[ni, nj]
    if s1 == s2:
        return 0.0

    nn = ((-1, 0), (1, 0), (0, -1), (0, 1))
    nnn = ((-1, -1), (-1, 1), (1, -1), (1, 1))

    dE = 0.0
    for di, dj in nn:
        ii, jj = i + di, j + dj
        if not (ii == ni and jj == nj):
            dE += J1 * (s2 - s1) * lattice[ii, jj]
        ii2, jj2 = ni + di, nj + dj
        if not (ii2 == i and jj2 == j):
            dE += J1 * (s1 - s2) * lattice[ii2, jj2]

    for di, dj in nnn:
        ii, jj = i + di, j + dj
        if not (ii == ni and jj == nj):
            dE += J2 * (s2 - s1) * lattice[ii, jj]
        ii2, jj2 = ni + di, nj + dj
        if not (ii2 == i and jj2 == j):
            dE += J2 * (s1 - s2) * lattice[ii2, jj2]

    return dE


@njit
def local_energy_change_flip(lattice, i, j, new_s, J1, J2, muB, muR):
    """
    ΔE for replacing lattice[i,j] with new_s.
    """
    old_s = lattice[i, j]
    if old_s == new_s:
        return 0.0

    ds = new_s - old_s
    nn_sum = (
        lattice[i - 1, j] + lattice[i + 1, j] + lattice[i, j - 1] + lattice[i, j + 1]
    )
    nnn_sum = (
        lattice[i - 1, j - 1]
        + lattice[i - 1, j + 1]
        + lattice[i + 1, j - 1]
        + lattice[i + 1, j + 1]
    )

    dE = J1 * ds * nn_sum + J2 * ds * nnn_sum

    # removing old species: costs +mu (penalises removal of favoured species)
    if old_s == -1:
        dE += muB
    if old_s == 1:
        dE += muR

    # adding new species: costs -mu (rewards addition of favoured species)
    if new_s == -1:
        dE -= muB
    if new_s == 1:
        dE -= muR

    return dE


@njit
def pick_different(old_s):
    """Pick uniformly from the two species that are not old_s."""
    r = np.random.randint(0, 2)
    if old_s == -1:
        return 0 if r == 0 else 1
    if old_s == 0:
        return -1 if r == 0 else 1
    return -1 if r == 0 else 0


@njit
def find_surface_site(Lx, Ly):
    """
    Pick uniformly from the four edge strips of the interior rectangle.
    Mathematica findEdgePosition: 1-based [4, rows-3] → 0-based [3, rows-4].
    """
    r_lo, r_hi = 3, Lx - 4
    c_lo, c_hi = 3, Ly - 4
    edge = np.random.randint(0, 4)
    if edge == 0:
        return r_lo, np.random.randint(c_lo, c_hi + 1)
    if edge == 1:
        return r_hi, np.random.randint(c_lo, c_hi + 1)
    if edge == 2:
        return np.random.randint(r_lo, r_hi + 1), c_lo
    return np.random.randint(r_lo, r_hi + 1), c_hi


@njit
def try_surface_insert(lattice, J1, J2, muB, muR, T):
    """One surface-band insertion (Mathematica grand-canonical block)."""
    Lx, Ly = lattice.shape
    i, j = find_surface_site(Lx, Ly)
    new_s = pick_different(lattice[i, j])
    dE = local_energy_change_flip(lattice, i, j, new_s, J1, J2, muB, muR)
    if dE <= 0 or np.random.rand() < np.exp(-dE / T):
        lattice[i, j] = new_s


@njit
def pick_valid_neighbor(i, j, i_lo, i_hi, j_lo, j_hi):
    valid_idx = np.empty(8, dtype=np.int64)
    n = 0

    for k in range(8):
        di = DIRECTIONS[k, 0]
        dj = DIRECTIONS[k, 1]

        ni, nj = i + di, j + dj
        if i_lo <= ni <= i_hi and j_lo <= nj <= j_hi:
            valid_idx[n] = k
            n += 1

    if n == 0:
        return i, j

    kk = valid_idx[np.random.randint(0, n)]
    return i + DIRECTIONS[kk, 0], j + DIRECTIONS[kk, 1]


@njit
def monte_carlo_step(
    lattice, vacancy_coords, n_vac, n_steps, T, J1, J2, muB, muR, mode
):
    """
    Main simulation loop. Maintains vacancy_coords in sync with lattice.
    """
    Lx, Ly = lattice.shape
    i_lo, i_hi = 3, Lx - 4
    j_lo, j_hi = 3, Ly - 4

    for _ in range(n_steps):
        # vacancy swap
        if n_vac > 0:
            vac_idx = np.random.randint(0, n_vac)
            i, j = vacancy_coords[vac_idx, 0], vacancy_coords[vac_idx, 1]
            ni, nj = pick_valid_neighbor(i, j, i_lo, i_hi, j_lo, j_hi)

            # compute energy change and accept / reject
            if ni != i or nj != j:
                dE = local_energy_change_swap(lattice, i, j, ni, nj, J1, J2)
                if dE <= 0 or np.random.rand() < np.exp(-dE / T):
                    lattice[i, j], lattice[ni, nj] = lattice[ni, nj], lattice[i, j]
                    vacancy_coords[vac_idx, 0] = ni
                    vacancy_coords[vac_idx, 1] = nj

        if mode == 0:
            pass
        elif mode == 1:
            i2 = np.random.randint(i_lo, i_hi + 1)
            j2 = np.random.randint(j_lo, j_hi + 1)
        elif mode == 2:
            i2, j2 = find_surface_site(Lx, Ly)

        old_s = lattice[i2, j2]
        new_s = pick_different(old_s)

        # compute energy change and accept / reject
        dE = local_energy_change_flip(lattice, i2, j2, new_s, J1, J2, muB, muR)
        if dE <= 0 or np.random.rand() < np.exp(-dE / T):
            lattice[i2, j2] = new_s

            if old_s == 0 and new_s != 0:
                # vacancy was consumed: remove from list
                for k in range(n_vac):
                    if vacancy_coords[k, 0] == i2 and vacancy_coords[k, 1] == j2:
                        vacancy_coords[k, 0] = vacancy_coords[n_vac - 1, 0]
                        vacancy_coords[k, 1] = vacancy_coords[n_vac - 1, 1]
                        n_vac -= 1
                        break

            elif old_s != 0 and new_s == 0:
                # new vacancy created: add to list
                if n_vac < vacancy_coords.shape[0]:
                    vacancy_coords[n_vac, 0] = i2
                    vacancy_coords[n_vac, 1] = j2
                    n_vac += 1

    return n_vac


def run_simulation(
    lattice, vacancy_coords, n_steps, T, J1, J2, muB=0.0, muR=0.0, mode=0
):
    """
    Run n_steps MC steps. Returns (new_lattice, new_vacancy_coords).

    Parameters
    ----------
    lattice         : 2-D int64 array  (from initialize_lattice or prior run)
    vacancy_coords  : (N,2) int64 array of vacancy positions (from initialize_lattice
                      or prior run_simulation)
    n_steps         : number of MC steps
    T               : temperature
    J1, J2          : NN and NNN coupling constants
    muB, muR        : chemical potentials (positive = favours that species)
    mode            : 0 = constant composition, 1 = grand canonical,
                      2 = surface grand canonical

    Returns
    -------
    lat  : updated lattice (copy)
    vac  : updated vacancy_coords (copy)
    """
    lat = lattice.copy()

    Lx, Ly = lat.shape
    max_vac = (Lx - 6) * (Ly - 6)
    vac = np.empty((max_vac, 2), dtype=np.int64)
    n = vacancy_coords.shape[0]
    vac[:n] = vacancy_coords

    n = monte_carlo_step(lat, vac, n, n_steps, T, J1, J2, muB, muR, mode)
    new_vac = vac[:n].copy()

    return lat, new_vac


def plot_lattice_disks(lattice, ax=None, radius=0.45, display_width=640, padding=60):
    nx, ny = lattice.shape
    patches = []
    colors = []

    if ax is None:
        cell_size = display_width / (ny - 6)
        height = (nx - 6) * cell_size + padding
        dpi = 72
        fig, ax = plt.subplots(figsize=(display_width / dpi, height / dpi), dpi=dpi)
    else:
        fig = ax.figure

    ax_divider = make_axes_locatable(ax)
    c_ax = ax_divider.append_axes("bottom", size="5%", pad="5%")
    c_ax.set_xlim(0, 1)
    c_ax.set_xticks([])
    c_ax.set_yticks([])
    for spine in c_ax.spines.values():
        spine.set_visible(False)

    fr, fb, fv = compute_fractions(lattice)
    bar_r = Rectangle((0, 0), fr, 1, color=(0.9, 0.2, 0.2), alpha=0.8)
    bar_b = Rectangle((fr, 0), fb, 1, color=(0.2, 0.3, 0.9), alpha=0.8)
    bar_v = Rectangle((fr + fb, 0), fv, 1, color=(0.85, 0.85, 0.85), alpha=0.55)

    c_ax.add_patch(bar_r)
    c_ax.add_patch(bar_b)
    c_ax.add_patch(bar_v)
    c_ax.set_title(f"Red: {fr:.4f} | Blue: {fb:.4f} | Vacancy: {fv:.4f}", fontsize=13)

    for i in range(3, nx - 3):
        for j in range(3, ny - 3):
            s = lattice[i, j]

            if s == 0:
                continue

            circle = Circle((j, i), radius=radius)
            patches.append(circle)

            if s == 1:
                colors.append((230 / 255, 51 / 255, 51 / 255))  # red
            elif s == -1:
                colors.append((51 / 255, 77 / 255, 230 / 255))  # blue

    collection = PatchCollection(
        patches, facecolor=colors, edgecolor=None, linewidth=0.3
    )
    ax.add_collection(collection)

    for i in range(3, nx - 3):
        for j in range(3, ny - 3):
            if lattice[i, j] == 0:
                circ = Circle(
                    (j, i),
                    radius=radius,
                    color=(217 / 255, 217 / 255, 217 / 255),
                    alpha=0.75,
                )
                ax.add_patch(circ)

    ax.set_xlim(2.5, ny - 3.5)
    ax.set_ylim(2.5, nx - 3.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")
    fig.tight_layout()

    return ax


def compute_fractions(lattice):
    interior = lattice[3:-3, 3:-3]
    total = interior.size

    n_red = np.sum(interior == 1)
    n_blue = np.sum(interior == -1)
    n_vac = np.sum(interior == 0)

    return n_red / total, n_blue / total, n_vac / total


def total_energy(lattice, J1, J2, interior_slice=(slice(3, -3), slice(3, -3))):
    lat = lattice[interior_slice]
    E = 0.0

    # nearest neighbours
    E += J1 * np.sum(lat[:-1, :] * lat[1:, :])
    E += J1 * np.sum(lat[:, :-1] * lat[:, 1:])

    # next-nearest neighbours
    E += J2 * np.sum(lat[:-1, :-1] * lat[1:, 1:])
    E += J2 * np.sum(lat[:-1, 1:] * lat[1:, :-1])

    return E


def setup_lattice_canvas(lattice, cell_size=12, padding=60, display_width=640):
    nx, ny = lattice.shape
    width = (ny - 6) * cell_size
    height = (nx - 6) * cell_size + padding

    pixel_ratio = 2

    multi_canvas = MultiCanvas(
        2, width=width * pixel_ratio, height=height * pixel_ratio
    )

    for canvas in multi_canvas:
        canvas.scale(pixel_ratio)

    multi_canvas.layout.width = f"{display_width}px"
    multi_canvas.layout.height = "auto"

    bg_canvas = multi_canvas[0]
    bg_canvas.fill_style = "#fdfdfd"
    bg_canvas.fill_rect(0, 0, width, height)

    return multi_canvas, cell_size, padding, width, height


def draw_lattice(canvas, lattice, cell_size, radius=0.45):
    interior = lattice[3:-3, 3:-3]
    nx, ny = interior.shape

    with hold_canvas(canvas):
        canvas.clear()
        for i in range(nx):
            for j in range(ny):
                val = interior[i, j]

                if val == 1:
                    canvas.fill_style = "rgb(230, 51, 51)"
                    canvas.global_alpha = 1.0
                elif val == -1:
                    canvas.fill_style = "rgb(51, 77, 230)"
                    canvas.global_alpha = 1.0
                else:
                    canvas.fill_style = "rgb(217, 217, 217)"
                    canvas.global_alpha = 0.75

                canvas.fill_arc(
                    j * cell_size + cell_size / 2,
                    i * cell_size + cell_size / 2,
                    cell_size * radius,
                    0,
                    2 * np.pi,
                )


def draw_bars(canvas, lattice, width, height, padding):
    fr, fb, fv = compute_fractions(lattice)
    bar_height = 15
    bar_y = height - padding + 25

    with hold_canvas(canvas):
        canvas.fill_style = "#fdfdfd"
        canvas.fill_rect(0, height - padding, width, padding)

        x = 0
        for frac, color in zip([fr, fb, fv], ["#e63333", "#334de6", "#d9d9d9"]):
            w = frac * width
            canvas.fill_style = color
            canvas.global_alpha = 0.8
            canvas.fill_rect(x, bar_y, w, bar_height)
            x += w

        canvas.global_alpha = 1.0
        canvas.fill_style = "black"
        canvas.font = "13px sans-serif"
        canvas.text_align = "center"
        canvas.fill_text(
            f"Red: {fr:.3f} | Blue: {fb:.3f} | Vacancy: {fv:.3f}", width / 2, bar_y - 8
        )


def interactive_kmc_canvas(
    lattice,
    vacancy_coords,
    step_fn,
    T,
    J1,
    J2,
    muB,
    muR,
    mode,
    steps_per_frame=250,
    interval=100,
    display_width=640,
    cell_size=12,
):
    multi_canvas, cell_size, padding, width, height = setup_lattice_canvas(
        lattice,
        cell_size=cell_size,
        display_width=display_width,
    )
    bg_layer, lat_layer = multi_canvas[0], multi_canvas[1]

    style = {"description_width": "initial"}
    ctrl_layout = Layout(width=f"{display_width // 2}px")

    J1_slider = FloatSlider(
        value=J1,
        min=-10.0,
        max=10.0,
        step=0.1,
        description="NN interaction",
        layout=ctrl_layout,
        style=style,
    )
    J2_slider = FloatSlider(
        value=J2,
        min=-10.0,
        max=10.0,
        step=0.1,
        description="NNN interaction",
        layout=ctrl_layout,
        style=style,
    )

    muB_slider = FloatSlider(
        value=muB,
        min=-10.0,
        max=10.0,
        step=0.1,
        description="Chemical potential blue",
        layout=ctrl_layout,
        style=style,
    )
    muR_slider = FloatSlider(
        value=muR,
        min=-10.0,
        max=10.0,
        step=0.1,
        description="Chemical potential red",
        layout=ctrl_layout,
        style=style,
    )

    T_slider = FloatSlider(
        value=T,
        min=0.05,
        max=10.0,
        step=0.05,
        description="Temperature",
        layout=ctrl_layout,
        style=style,
    )

    mode_dropdown = Dropdown(
        options=[
            ("Canonical", 0),
            ("Grand Canonical", 1),
            ("Surface Grand Canonical", 2),
        ],
        value=mode,
        description="Ensemble",
        layout=ctrl_layout,
        style=style,
    )

    steps_slider = IntSlider(
        value=steps_per_frame,
        min=1,
        max=1000,
        description="Steps/frame",
        layout=ctrl_layout,
    )
    interval_slider = IntSlider(
        value=interval, min=10, max=500, description="Speed (ms)", layout=ctrl_layout
    )

    play = Play(
        interval=interval,
        value=0,
        min=0,
        max=9999,
        step=1,
        layout=ctrl_layout,
        repeat=True,
    )
    jslink((interval_slider, "value"), (play, "interval"))

    is_busy = False

    def update_view():
        draw_lattice(lat_layer, lattice, cell_size)
        draw_bars(bg_layer, lattice, width, height, padding)

    def on_step(change):
        nonlocal lattice, vacancy_coords, is_busy
        if is_busy:
            return
        try:
            is_busy = True
            lattice, vacancy_coords = step_fn(
                lattice,
                vacancy_coords,
                steps_slider.value,
                T_slider.value,
                J1_slider.value,
                J2_slider.value,
                muB_slider.value,
                muR_slider.value,
                mode_dropdown.value,
            )
            update_view()
        finally:
            is_busy = False

    reset_btn = Button(description="Reset Lattice", icon="refresh", layout=ctrl_layout)

    initial_lattice = lattice.copy()
    initial_vacs = vacancy_coords.copy()

    def reset(_):
        nonlocal lattice, vacancy_coords
        lattice, vacancy_coords = initial_lattice.copy(), initial_vacs.copy()
        update_view()

    reset_btn.on_click(reset)
    play.observe(on_step, names="value")

    full_ui = VBox(
        [
            HBox([play, reset_btn]),
            HBox([J1_slider, J2_slider]),
            HBox([muB_slider, muR_slider]),
            HBox([T_slider, mode_dropdown]),
            HBox([steps_slider, interval_slider]),
            multi_canvas,
        ]
    )

    update_view()
    display(full_ui)
