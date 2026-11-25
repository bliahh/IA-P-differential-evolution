"""Cursa pinguinilor: evolutia diferentiala cauta cea mai rapida trasa pe un circuit de gheata.

Un individ = o trasa completa. Genele spun cat de la stanga (-1) sau la dreapta (1) trece pinguinul
in GENE puncte ale circuitului. Intre puncte, trasa e un B-spline cubic periodic: ramane mereu pe pista.
Fitness (il minimizam): timpul pe tur, in secunde.

Rulare: python penguin_race.py          (fereastra)
        python penguin_race.py --check  (verificare fara fereastra)
Taste:  SPATIU pauza, C cursa / inapoi la evolutie, R alt seed, SUS/JOS viteza, ESC iesire
"""
import math
import sys
from pathlib import Path

import numpy as np
import pygame

sys.path.insert(0, str(Path(__file__).parent / "common"))  # algoritmul face `from Chromosome import ...`
from DifferentialEvolution import DifferentialEvolution  # noqa: E402

# circuitul (metri) si fizica pinguinului care aluneca pe burta
CIRCUIT = [(-50, -30), (-20, -36), (15, -34), (48, -36), (60, -14), (38, -2), (12, -12), (-4, 4), (14, 20),
           (46, 14), (60, 32), (36, 46), (0, 40), (-22, 26), (-30, 44), (-56, 38), (-60, 12), (-42, -6)]
JUMATATE_PISTA = 5.0
MARGINE = 0.8          # cat spatiu lasa pinguinul pana la marginea pistei
PUNCTE = 300           # in cate puncte calculam trasa
A_LATERAL = 5.0        # aderenta in viraj (m/s^2)
A_ACCEL, A_FRANA = 2.5, 4.0
V_MAX = 16.0

# evolutia diferentiala
GENE, NP, GENERATII, F, CR, SEED = 20, 40, 200, 0.5, 0.9, 42


def bspline(n, m):
    """Matricea m x n: valorile a n puncte de control -> m puncte pe curba inchisa."""
    t = np.arange(m) * n / m
    i = t.astype(int)
    u = t - i
    ponderi = [(1 - u) ** 3 / 6, (3 * u ** 3 - 6 * u ** 2 + 4) / 6,
               (-3 * u ** 3 + 3 * u ** 2 + 3 * u + 1) / 6, u ** 3 / 6]
    b = np.zeros((m, n))
    for k, w in enumerate(ponderi):
        b[np.arange(m), (i - 1 + k) % n] = w
    return b


def build_center():
    """Mijlocul pistei, cu puncte la distante egale, si normala in fiecare punct."""
    fin = bspline(len(CIRCUIT), 4000) @ np.array(CIRCUIT, dtype=float)
    inchis = np.vstack([fin, fin[:1]])
    s = np.r_[0, np.cumsum(np.hypot(*np.diff(inchis, axis=0).T))]
    tinta = np.linspace(0, s[-1], PUNCTE, endpoint=False)
    c = np.c_[np.interp(tinta, s, inchis[:, 0]), np.interp(tinta, s, inchis[:, 1])]
    d = np.roll(c, -1, 0) - np.roll(c, 1, 0)
    d /= np.hypot(*d.T)[:, None]
    return c, np.c_[d[:, 1], -d[:, 0]]


CENTRU, NORMALA = build_center()
BAZA = bspline(GENE, PUNCTE)


def build_line(gene):
    return CENTRU + NORMALA * (BAZA @ gene * (JUMATATE_PISTA - MARGINE))[:, None]


def curvature(p):
    d1 = (np.roll(p, -1, 0) - np.roll(p, 1, 0)) / 2
    d2 = np.roll(p, -1, 0) - 2 * p + np.roll(p, 1, 0)
    return np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / np.hypot(*d1.T) ** 3


def speed_profile(p):
    """Viteza in fiecare punct si durata fiecarui segment, pentru un tur lansat."""
    n = len(p)
    ds = np.hypot(*(np.roll(p, -1, 0) - p).T)
    v2 = np.minimum(A_LATERAL / np.maximum(curvature(p), 1e-9), V_MAX ** 2)
    start = int(v2.argmin())  # in cel mai lent punct viteza e sigur atinsa, deci ajunge o trecere in fiecare sens
    w, d = np.roll(v2, -start).tolist(), np.roll(ds, -start).tolist()
    w.append(w[0])
    for i in range(n):
        w[i + 1] = min(w[i + 1], w[i] + 2 * A_ACCEL * d[i])
    for i in range(n - 1, -1, -1):
        w[i] = min(w[i], w[i + 1] + 2 * A_FRANA * d[i])
    v = np.sqrt(np.roll(w[:-1], start))
    return v, 2 * ds / (v + np.roll(v, -1))


def lap_time(gene):
    return float(speed_profile(build_line(gene))[1].sum())


def new_de(seed=SEED):
    return DifferentialEvolution(NP, lap_time, GENE, [1] * GENE, [-1] * GENE, GENERATII, F, CR, seed)


def check():
    # pe un cerc viteza e constanta, deci timpul se stie exact
    a = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    cerc = 20 * np.c_[np.cos(a), np.sin(a)]
    assert abs(speed_profile(cerc)[1].sum() - 2 * np.pi * 20 / math.sqrt(A_LATERAL * 20)) < 0.01
    # limitele de acceleratie si franare sunt respectate pe o trasa oarecare
    rng = np.random.default_rng(0)
    p = build_line(rng.uniform(-1, 1, GENE))
    v, dt = speed_profile(p)
    a_lung = (np.roll(v, -1) - v) / dt
    assert a_lung.max() < A_ACCEL + 1e-6 and a_lung.min() > -A_FRANA - 1e-6
    assert (v ** 2 * curvature(p)).max() < A_LATERAL + 1e-6
    # pista nu se incaleca pe interiorul virajelor
    assert (curvature(CENTRU) * JUMATATE_PISTA).max() < 1, curvature(CENTRU).max()

    # in cursa conteaza cine e in fata acum, nu cine va castiga
    bun_la_inceput, bun_la_final = np.array([0, 1, 9.0]), np.array([0, 4, 5.0])
    assert race_rank(2, bun_la_inceput) < race_rank(2, bun_la_final)
    assert race_rank(6, bun_la_final) < race_rank(6, bun_la_inceput)

    centru = lap_time(np.zeros(GENE))
    for seed in (1, 2, 3):
        cel_mai_bun = new_de(seed).evolve()[1]
        rng = np.random.default_rng(seed)
        aleator = min(lap_time(rng.uniform(-1, 1, GENE)) for _ in range(NP * (GENERATII + 1)))
        print(f"seed {seed}: mijloc {centru:.2f} s, aleator {aleator:.2f} s, DE {cel_mai_bun:.2f} s")
        assert cel_mai_bun < aleator and cel_mai_bun < centru


# ---------------- fereastra ----------------

W, H, HARTA = 1200, 720, 860   # HARTA = latimea zonei cu circuitul; restul e panoul din dreapta
SCALA = 6.2                    # pixeli pe metru
FUNDAL, GHEATA, BORDURA = (203, 222, 235), (240, 248, 253), (120, 160, 190)
PANOU, TEXT, GRI = (15, 23, 42), (226, 232, 240), (148, 163, 184)
CULORI = [(148, 163, 184), (239, 68, 68), (249, 115, 22), (168, 85, 247), (34, 197, 94)]  # concurentii din cursa


def to_screen(p):
    mijloc = (CENTRU.min(0) + CENTRU.max(0)) / 2
    return (np.asarray(p) - mijloc) * (SCALA, -SCALA) + (HARTA / 2, H / 2)


def make_penguin(fular):
    """Pinguin pe burta, vazut de sus, cu ciocul spre dreapta."""
    s = pygame.Surface((34, 26), pygame.SRCALPHA)
    pygame.draw.ellipse(s, (51, 65, 85), (8, 0, 12, 26))      # aripi
    pygame.draw.ellipse(s, (20, 24, 34), (0, 5, 26, 16))      # corp
    pygame.draw.polygon(s, (251, 146, 60), [(27, 10), (34, 13), (27, 16)])  # cioc
    pygame.draw.circle(s, (20, 24, 34), (24, 13), 6)          # cap
    pygame.draw.rect(s, fular, (16, 6, 4, 14))                # fular
    return s


class Evolution:
    def __init__(self, seed):
        self.de = new_de(seed)
        self.seed, self.gen = seed, 0
        self.gene = self.vechi = self.read_genes()
        self.profile = [speed_profile(build_line(g)) for g in self.gene]
        self.campioni, self.timpi, self.medii = [], [], []
        self.record()

    def read_genes(self):
        return np.array([c.genes for c in self.de.population])

    def record(self):
        fit = np.array([c.val_fitness_function for c in self.de.population])
        self.best = int(fit.argmin())
        self.campioni.append(self.gene[self.best].copy())
        self.timpi.append(float(fit.min()))
        self.medii.append(float(fit.mean()))

    def step(self):
        self.vechi = self.gene
        self.de.selection(minimize=True)
        self.gene = self.read_genes()
        self.gen += 1
        for i in np.flatnonzero((self.gene != self.vechi).any(axis=1)):
            self.profile[i] = speed_profile(build_line(self.gene[i]))
        self.record()


class Race:
    """Campionii catorva generatii si un pinguin care merge pe mijloc, toti deodata, un tur."""

    def __init__(self, evo):
        generatii = sorted({0, evo.gen // 8, evo.gen // 3, evo.gen})
        concurenti = [("mijlocul pistei", np.zeros(GENE))] + [(f"generatia {g}", evo.campioni[g]) for g in generatii]
        self.t, self.concurenti = 0.0, []
        for (nume, gene), culoare in zip(concurenti, CULORI[:len(concurenti) - 1] + CULORI[-1:]):
            p = build_line(gene)
            cumulat = np.r_[0, np.cumsum(speed_profile(p)[1])]
            self.concurenti.append((nume, culoare, p, cumulat, make_penguin(culoare)))


def race_rank(t, cumulat):
    """Cheia de clasament: intai cei ajunsi, dupa timp; apoi ceilalti, dupa cat au parcurs."""
    if cumulat[-1] <= t:
        return 0, cumulat[-1]
    return 1, -np.interp(t, cumulat, np.arange(len(cumulat)))


def point_on(p, u):
    """Pozitia si directia la indexul real u pe trasa p."""
    i = int(u) % len(p)
    a, b = p[i], p[(i + 1) % len(p)]
    return a + (u - int(u)) * (b - a), b - a


def draw_penguin(ecran, sprite, p, u):
    poz, d = point_on(p, u)
    rotit = pygame.transform.rotate(sprite, math.degrees(math.atan2(d[1], d[0])))
    ecran.blit(rotit, rotit.get_rect(center=tuple(to_screen(poz))))


def draw_chart(ecran, font, evo, x, y, w, h):
    pygame.draw.rect(ecran, (30, 41, 59), (x, y, w, h), border_radius=6)
    lo, hi = min(evo.timpi) - 0.3, max(evo.medii) + 0.3
    for valori, culoare in ((evo.medii, (251, 146, 60)), (evo.timpi, (56, 189, 248))):
        if len(valori) > 1:
            pygame.draw.lines(ecran, culoare, False, [
                (x + 8 + k / GENERATII * (w - 16), y + h - 8 - (v - lo) / (hi - lo) * (h - 16))
                for k, v in enumerate(valori)], 2)
    for k, (nume, culoare) in enumerate((("cel mai bun", (56, 189, 248)), ("media populatiei", (251, 146, 60)))):
        ecran.blit(font.render(nume, True, culoare), (x + w - 150, y + 8 + 18 * k))


def main():
    pygame.init()
    ecran = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Cursa pinguinilor: evolutie diferentiala")
    mare, font, mic = (pygame.font.Font(None, m) for m in (34, 24, 20))
    ceas = pygame.time.Clock()

    # circuitul nu se schimba, deci il desenam o singura data
    fundal = pygame.Surface((HARTA, H))
    fundal.fill(FUNDAL)
    st, dr = (to_screen(CENTRU + semn * JUMATATE_PISTA * NORMALA) for semn in (-1, 1))
    for i in range(PUNCTE):
        j = (i + 1) % PUNCTE
        pygame.draw.polygon(fundal, GHEATA, [st[i], st[j], dr[j], dr[i]])
    for margine in (st, dr):
        pygame.draw.lines(fundal, BORDURA, True, margine, 3)
    pygame.draw.line(fundal, (30, 41, 59), st[0], dr[0], 5)
    strat = pygame.Surface((HARTA, H), pygame.SRCALPHA)  # trasele populatiei, transparente

    sprite_pop, sprite_best = make_penguin((100, 116, 139)), make_penguin((34, 197, 94))
    evo, cursa = Evolution(SEED), None
    start = np.arange(NP) * PUNCTE / NP  # pinguinii pornesc rasfirati pe circuit, ca sa nu se acopere
    poz = start.copy()                   # unde e fiecare pe trasa lui (index real)
    durata_gen, trecut = 0.25, 0.0
    pauza = False

    while True:
        dt = min(ceas.tick(60) / 1000, 0.05)
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                pygame.quit()
                return
            if e.type != pygame.KEYDOWN:
                continue
            if e.key == pygame.K_SPACE:
                pauza = not pauza
            elif e.key == pygame.K_c:
                cursa = None if cursa else Race(evo)
            elif e.key == pygame.K_r:
                evo, cursa, trecut, poz = Evolution(evo.seed + 1), None, 0.0, start.copy()
            elif e.key == pygame.K_UP:
                durata_gen = max(0.05, durata_gen / 2)
            elif e.key == pygame.K_DOWN:
                durata_gen = min(2.0, durata_gen * 2)

        if not pauza and cursa:
            cursa.t += 2 * dt  # cursa ruleaza de 2 ori mai repede decat in realitate
        elif not pauza:
            trecut += dt
            if trecut >= durata_gen and evo.gen < GENERATII:
                evo.step()
                trecut = 0.0
                if evo.gen == GENERATII:
                    cursa = Race(evo)  # evolutia s-a terminat: urmeaza cursa
            for i, (_, durate) in enumerate(evo.profile):
                poz[i] = (poz[i] + dt / durate[int(poz[i]) % PUNCTE]) % PUNCTE

        ecran.fill(PANOU)
        ecran.blit(fundal, (0, 0))
        x = HARTA + 24
        if cursa:
            for nume, culoare, p, cumulat, sprite in cursa.concurenti:
                pygame.draw.aalines(ecran, culoare, True, to_screen(p))
            for nume, culoare, p, cumulat, sprite in cursa.concurenti:
                draw_penguin(ecran, sprite, p, np.interp(cursa.t, cumulat, np.arange(PUNCTE + 1)))
            ecran.blit(mare.render("Cursa (viteza x2)", True, TEXT), (x, 24))
            ecran.blit(font.render(f"timp: {min(cursa.t, cursa.concurenti[0][3][-1]):.1f} s", True, GRI), (x, 60))
            ordine = sorted(cursa.concurenti, key=lambda c: race_rank(cursa.t, c[3]))
            for k, (nume, culoare, p, cumulat, sprite) in enumerate(ordine):
                sosit = cumulat[-1] <= cursa.t
                pygame.draw.circle(ecran, culoare, (x + 8, 112 + 34 * k), 8)
                ecran.blit(font.render(f"{k + 1}. {nume}" + (f": {cumulat[-1]:.2f} s" if sosit else ""), True, TEXT),
                           (x + 26, 104 + 34 * k))
        else:
            t = min(trecut / durata_gen, 1.0) if evo.gen < GENERATII else 1.0
            trase = [build_line(g) for g in evo.vechi + t * (evo.gene - evo.vechi)]
            strat.fill((0, 0, 0, 0))
            for p in trase:
                pygame.draw.lines(strat, (71, 85, 105, 70), True, to_screen(p), 1)
            ecran.blit(strat, (0, 0))
            pygame.draw.lines(ecran, (34, 197, 94), True, to_screen(trase[evo.best]), 3)
            for i in [*(k for k in range(NP) if k != evo.best), evo.best]:
                draw_penguin(ecran, sprite_best if i == evo.best else sprite_pop, trase[i], poz[i])
            ecran.blit(mare.render(f"Generatia {evo.gen} / {GENERATII}", True, TEXT), (x, 24))
            for k, linie in enumerate((f"cel mai bun tur: {evo.timpi[-1]:.2f} s",
                                       f"media populatiei: {evo.medii[-1]:.2f} s",
                                       f"pe mijlocul pistei: {lap_time(np.zeros(GENE)):.2f} s",
                                       f"populatie {NP}, gene {GENE}, F {F}, CR {CR}, seed {evo.seed}")):
                ecran.blit((font if k < 3 else mic).render(linie, True, GRI), (x, 64 + 26 * k))
        draw_chart(ecran, mic, evo, x, 320, W - x - 20, 220)
        ecran.blit(mic.render("timpul pe tur (s) pe generatii", True, GRI), (x, 546))
        for k, linie in enumerate(("SPATIU pauza" + (" (activa)" if pauza else ""),
                                   "C " + ("inapoi la evolutie" if cursa else "cursa intre campioni"),
                                   "R alt seed", "SUS / JOS viteza evolutiei", "ESC iesire")):
            ecran.blit(mic.render(linie, True, GRI), (x, 586 + 22 * k))
        pygame.display.flip()


if __name__ == "__main__":
    check() if "--check" in sys.argv else main()
