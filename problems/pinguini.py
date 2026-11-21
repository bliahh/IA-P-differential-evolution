"""Pinguini pe rampă (Petrică Avornicesei).

Un pinguin coboară pe o rampă din vârful unui deal, zboară, aterizează și alunecă pe burtă
(tobogganing) pe un relief cu movile, văi și copci. Genele: [unghiul rampei (grade), impulsul inițial (m/s)].
Fitness: distanța finală față de start (o maximizăm). Nu există formulă: fitness-ul vine din simulare.
"""
import math
from functools import lru_cache

import gradio as gr
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from DifferentialEvolution import DifferentialEvolution  # noqa: E402  (common/ e pus în sys.path de app.py)

G = 9.81
DX = 0.5            # rezoluția terenului (m)
LUNGIME = 400.0     # lungimea hărții (m)
H_DEAL = 40.0       # înălțimea dealului de start (m)
LATIME_DEAL = 70.0  # cât de lină e panta dealului (m)
H_RAMPA = 10.0      # cât coboară pinguinul pe rampă până la buză (m)
MU_RAMPA = 0.05     # frecarea pe rampă
DRAG = 0.003        # rezistența aerului: a = -DRAG·|v|·v
MU_GHEATA = 0.08    # frecarea la alunecarea pe burtă
V_SIGUR = 14.0     # viteza perpendiculară pe sol peste care pinguinul se lovește și nu mai alunecă (m/s)
LIM_JOS = [-10.0, 0.0]   # [unghi, impuls]
LIM_SUS = [60.0, 20.0]


@lru_cache(maxsize=16)
def terrain(seed):
    """Relief generat din seed: dealul de start, movile/văi aleatoare și două copci."""
    rng = np.random.default_rng(seed)
    x = np.arange(0.0, LUNGIME + DX, DX)
    y = H_DEAL * (1 - np.tanh(x / LATIME_DEAL))
    for _ in range(6):
        c, w, h = rng.uniform(60, LUNGIME - 20), rng.uniform(8, 30), rng.uniform(-8, 10)
        y += h * np.exp(-((x - c) / w) ** 2)
    apa = np.zeros_like(x, dtype=bool)
    for _ in range(3):
        c, w = rng.uniform(70, LUNGIME - 30), rng.uniform(6, 18)
        apa |= np.abs(x - c) < w / 2
    y[apa] -= 2.0
    return x, y, apa, y.tolist(), apa.tolist()


def simulate(gene, seed, traseu=False):
    """Întoarce (distanța, motivul opririi, traseu [(x, y)] dacă traseu=True).

    Pinguinul alternează între zbor (gravitație + rezistența aerului) și alunecare pe burtă (gravitație +
    frecare). Decolează singur când terenul „fuge” de sub el (creasta unei movile, marginea unei copci),
    deci poate sări peste copci dacă are viteză. La fiecare aterizare pierde viteza perpendiculară pe sol;
    dacă aceasta depășește V_SIGUR, se lovește și rămâne acolo.
    """
    _, _, _, ys, apa = terrain(seed)
    n = len(ys) - 1

    def idx(x):
        return min(max(int(x / DX), 0), n - 1)

    def height(x):
        i = idx(x)
        return ys[i] + (ys[i + 1] - ys[i]) * (x / DX - i)

    unghi, impuls = math.radians(gene[0]), gene[1]
    v = math.sqrt(impuls ** 2 + 2 * G * H_RAMPA * (1 - MU_RAMPA))
    x, y = 0.0, ys[0] + 0.5
    vx, vy = v * math.cos(unghi), v * math.sin(unghi)
    u = 0.0                # viteza de-a lungul solului, cât timp alunecă
    in_aer = True
    pts = [(x, y)] if traseu else None
    dt = 0.01

    for _ in range(100_000):
        if in_aer:
            sp = math.hypot(vx, vy)
            vx -= DRAG * sp * vx * dt
            vy -= (G + DRAG * sp * vy) * dt
            x += vx * dt
            y += vy * dt
            if x >= LUNGIME:
                return LUNGIME, "a zburat dincolo de hartă", pts
            if y <= height(x):                       # aterizare
                i = idx(x)
                if apa[i]:
                    return x, "a căzut într-o copcă", pts
                a = math.atan((ys[i + 1] - ys[i]) / DX)
                v_perp = vx * math.sin(a) - vy * math.cos(a)
                if v_perp > V_SIGUR:
                    return x, f"s-a lovit la aterizare ({v_perp:.1f} m/s)", pts
                u = vx * math.cos(a) + vy * math.sin(a)
                y = height(x)
                in_aer = False
        else:
            i = idx(x)
            s = (ys[i + 1] - ys[i]) / DX
            cos_a = 1 / math.sqrt(1 + s * s)
            sin_a = s * cos_a
            static = abs(sin_a) <= MU_GHEATA * cos_a
            if abs(u) < 0.05 and static:
                return x, "s-a oprit pe gheață", pts
            u_nou = u - G * sin_a * dt - math.copysign(MU_GHEATA * G * cos_a * dt, u)
            if u * u_nou < 0 and static:
                return x, "s-a oprit pe gheață", pts
            u = u_nou
            vx, vy = u * cos_a, u * sin_a
            x += vx * dt
            y_liber = y + vy * dt - 0.5 * G * dt * dt   # unde ar fi dacă ar zbura liber
            if y_liber > height(x) + 1e-3:            # terenul coboară mai repede decât cade el: decolează
                y, in_aer = y_liber, True
            else:
                y = height(x)
            if x >= LUNGIME:
                return LUNGIME, "a alunecat până la capătul hărții", pts
            if x <= 0:
                return 0.0, "a alunecat înapoi la start", pts
        if traseu:
            pts.append((x, y))
    return x, "s-a oprit pe gheață", pts


@lru_cache(maxsize=8)
def fitness_landscape(seed, pasi=60):
    """Fitness-ul pe o grilă unghi × impuls (căutarea exhaustivă, pentru comparație cu DE)."""
    u = np.linspace(LIM_JOS[0], LIM_SUS[0], pasi)
    p = np.linspace(LIM_JOS[1], LIM_SUS[1], pasi)
    z = np.array([[simulate((a, b), seed)[0] for a in u] for b in p])
    return u, p, z


# ---------------- desene ----------------

def _draw_terrain(ax, seed):
    x, y, apa, _, _ = terrain(seed)
    ax.fill_between(x, y.min() - 5, y, color="#dfe9f3")
    ax.plot(x, y, color="#5a7184", lw=1)
    ax.fill_between(x, y.min() - 5, y + 1.5, where=apa, color="#3b82c4", alpha=0.8)
    lx = H_RAMPA / math.tan(math.radians(35))
    ax.plot([-lx, 0], [y[0] + 0.5 + H_RAMPA, y[0] + 0.5], color="#8b5a2b", lw=3)
    ax.set_xlim(-lx - 5, LUNGIME)
    ax.set_xlabel("distanță (m)")
    ax.set_ylabel("înălțime (m)")


def plot_trajectories(seed, gene_list, etichete):
    fig, ax = plt.subplots(figsize=(10, 3.6))
    _draw_terrain(ax, seed)
    culori = plt.cm.viridis(np.linspace(0.15, 0.95, len(gene_list)))
    for g, e, c in zip(gene_list, etichete, culori):
        d, _, pts = simulate(g, seed, traseu=True)
        px, py = zip(*pts)
        ax.plot(px, py, color=c, lw=1.6, label=f"{e}: {d:.1f} m")
        ax.plot(px[-1], py[-1], "o", color=c, ms=6)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def plot_convergence(cel_mai_bun, medie):
    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.plot(cel_mai_bun, label="cel mai bun")
    ax.plot(medie, label="media populației", alpha=0.7)
    ax.set_xlabel("generația")
    ax.set_ylabel("distanța (m)")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_landscape(seed, pop_start, pop_final, best):
    u, p, z = fitness_landscape(seed)
    fig, ax = plt.subplots(figsize=(5, 3.4))
    im = ax.pcolormesh(u, p, z, cmap="magma", shading="auto")
    fig.colorbar(im, ax=ax, label="distanța (m)")
    ax.scatter(pop_start[:, 0], pop_start[:, 1], s=12, c="white", marker="x", label="populația inițială")
    ax.scatter(pop_final[:, 0], pop_final[:, 1], s=14, c="#22d3ee", label="populația finală")
    ax.scatter([best[0]], [best[1]], s=120, c="lime", marker="*", label="cel mai bun")
    ax.set_xlabel("unghiul rampei (°)")
    ax.set_ylabel("impuls (m/s)")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    return fig


# ---------------- logica tabului ----------------

def run_de(seed_teren, np_, generatii, f, cr, seed_de):
    seed_teren, np_, generatii = int(seed_teren), int(np_), int(generatii)
    de = DifferentialEvolution(np_, lambda g: simulate(g, seed_teren)[0], 2,
                               LIM_SUS, LIM_JOS, generatii, f, cr, int(seed_de))
    populatii = [np.array([c.genes for c in de.population])]
    fit = [np.array([c.val_fitness_function for c in de.population])]
    for _ in range(generatii):
        de.selection(minimize=False)
        populatii.append(np.array([c.genes for c in de.population]))
        fit.append(np.array([c.val_fitness_function for c in de.population]))

    campioni = [populatii[k][fit[k].argmax()] for k in range(len(fit))]
    momente = sorted({0, generatii // 4, generatii // 2, generatii})
    best = campioni[-1]
    d, motiv, _ = simulate(best, seed_teren)
    rezumat = (f"**Cel mai bun pinguin:** unghi {best[0]:.1f}°, impuls {best[1]:.1f} m/s → "
               f"**{d:.1f} m** ({motiv}). Evaluări de fitness: {np_ * (generatii + 1)} "
               f"(grila din dreapta are {fitness_landscape(seed_teren)[2].size}).")
    return (plot_trajectories(seed_teren, [campioni[k] for k in momente], [f"gen {k}" for k in momente]),
            plot_convergence([x.max() for x in fit], [x.mean() for x in fit]),
            plot_landscape(seed_teren, populatii[0], populatii[-1], best),
            rezumat)


def manual_launch(seed_teren, unghi, impuls):
    d, motiv, _ = simulate((unghi, impuls), int(seed_teren))
    return plot_trajectories(int(seed_teren), [(unghi, impuls)], ["lansarea ta"]), f"**{d:.1f} m** ({motiv})"


def build_tab():
    gr.Markdown("### Pinguini pe rampă\nFiecare pinguin alege **unghiul rampei** și **impulsul** cu care pornește. "
                "Zboară, aterizează (dacă aterizează prea abrupt se lovește) și alunecă pe burtă. "
                "Scopul: să ajungă cât mai departe. Copcile (albastru) îl opresc.")
    with gr.Row():
        seed_teren = gr.Number(value=7, precision=0, label="Seed teren")
        np_ = gr.Slider(4, 100, value=20, step=1, label="Mărimea populației (NP)")
        generatii = gr.Slider(1, 200, value=40, step=1, label="Generații")
        f = gr.Slider(0.1, 1.5, value=0.8, step=0.05, label="F")
        cr = gr.Slider(0.0, 1.0, value=0.9, step=0.05, label="CR")
        seed_de = gr.Number(value=42, precision=0, label="Seed DE")
    buton = gr.Button("Rulează evoluția diferențială", variant="primary")
    rezumat = gr.Markdown()
    traiectorii = gr.Plot(label="Cel mai bun pinguin pe parcursul evoluției")
    with gr.Row():
        convergenta = gr.Plot(label="Convergență")
        harta = gr.Plot(label="Peisajul de fitness (unghi × impuls)")
    buton.click(run_de, [seed_teren, np_, generatii, f, cr, seed_de],
                [traiectorii, convergenta, harta, rezumat])

    with gr.Accordion("Încearcă tu o lansare", open=False):
        with gr.Row():
            unghi = gr.Slider(LIM_JOS[0], LIM_SUS[0], value=30, step=0.5, label="Unghiul rampei (°)")
            impuls = gr.Slider(LIM_JOS[1], LIM_SUS[1], value=10, step=0.5, label="Impuls (m/s)")
        rezultat = gr.Markdown()
        plot_manual = gr.Plot()
        for c in (unghi, impuls, seed_teren):
            c.change(manual_launch, [seed_teren, unghi, impuls], [plot_manual, rezultat])


if __name__ == "__main__":  # verificare rapidă a simulării
    d, motiv, pts = simulate((30, 10), 7, traseu=True)
    assert 0 < d <= LUNGIME and len(pts) > 10, (d, motiv)
    u, p, z = fitness_landscape(7)
    i, j = np.unravel_index(z.argmax(), z.shape)
    print(f"optim pe grilă: unghi {u[j]:.1f}°, impuls {p[i]:.1f} → {z.max():.1f} m; "
          f"distanțe {z.min():.1f}–{z.max():.1f}; % puncte în maxim: {(z > z.max() - 1).mean():.1%}")
