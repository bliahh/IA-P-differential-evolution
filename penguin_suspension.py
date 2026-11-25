"""Suspensia masinii cu pilot pinguin: evolutia diferentiala regleaza arcurile si amortizoarele.

Model: jumatate de masina vazuta din lateral (caroserie care urca si se inclina + doua roti), pe un drum cu
denivelari, la viteza constanta. Pozitiile sunt masurate fata de echilibrul static.
Genele (6): rigiditatea arcului fata si spate, amortizarea la comprimare si la destindere, fata si spate.
Fitness (il minimizam): cat e zguduit pilotul + cat variaza apasarea rotilor pe drum.

Rulare: python penguin_suspension.py          (fereastra)
        python penguin_suspension.py --check  (verificare fara fereastra)
Taste:  SPATIU pauza, R alt seed, SUS/JOS viteza evolutiei, ESC iesire
"""
import math
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pygame

sys.path.insert(0, str(Path(__file__).parent / "common"))  # algoritmul face `from Chromosome import ...`
from DifferentialEvolution import DifferentialEvolution  # noqa: E402

# masina (SI)
M, I = 600.0, 1000.0           # masa si momentul de inertie ale caroseriei
A, B = 1.2, 1.4                # distanta de la centrul de masa la puntea fata / spate
M_ROATA, K_PNEU = 40.0, 180e3
SCAUN = -0.3                   # pozitia pilotului fata de centrul de masa
CURSA, K_TAMPON = 0.09, 300e3  # cat se poate comprima sau intinde suspensia, apoi loveste tamponul
SARCINA = 60 * 9.81             # N pe punte, pilotul si bagajul: cu un arc moale, masina se lasa si ramane fara cursa
VITEZA, LUNGIME = 14.0, 150.0  # m/s, m
DT = 0.004
G = 9.81
# apasarea statica pe fiecare punte
N_FATA = G * (M * B / (A + B) + M_ROATA)
N_SPATE = G * (M * A / (A + B) + M_ROATA)

# evolutia diferentiala
NUME = ["arc fata", "arc spate", "compr. fata", "destind. fata", "compr. spate", "destind. spate"]
LIM_JOS = [5e3, 5e3, 100, 100, 100, 100]        # N/m, N/m, apoi N*s/m
LIM_SUS = [80e3, 80e3, 8e3, 8e3, 8e3, 8e3]
OBISNUIT = [20e3, 20e3, 1500, 1500, 1500, 1500]  # un reglaj obisnuit, pentru comparatie
PONDERE_APASARE = 5.0
NP, GENERATII, F, CR, SEED = 30, 60, 0.6, 0.9, 42


def road(x):
    """Inaltimea drumului (m)."""
    x = np.asarray(x, dtype=float)
    h = 0.004 * np.sin(2.1 * x) + 0.003 * np.sin(3.7 * x + 1) + 0.002 * np.sin(6.3 * x + 2)  # asfalt vechi
    h += np.where((x > 14) & (x < 28), 0.015 * np.sin(2 * np.pi * (x - 14) / 1.4), 0)        # valurele
    for centru, lat, inalt in ((40, 1.6, 0.09), (52, 1.6, 0.09), (108, 1.0, -0.07), (116, 1.4, -0.08),
                               (128, 3.0, 0.14)):                                            # praguri si gropi
        h += np.where(np.abs(x - centru) < lat / 2, inalt * np.cos(np.pi * (x - centru) / lat) ** 2, 0)
    h += np.where((x > 64) & (x < 100), 0.10 * (1 - np.cos(2 * np.pi * (x - 64) / 12)) / 2, 0)  # valuri lungi
    return h * np.clip(x / 5, 0, 1) * np.clip((LUNGIME - x) / 5, 0, 1)


PASI = int(LUNGIME / VITEZA / DT)
X_FATA = VITEZA * DT * np.arange(PASI)
DRUM_FATA, DRUM_SPATE = road(X_FATA), road(X_FATA - (A + B))


def simulate(gene, dt=DT, drum=None):
    """Simuleaza tot drumul. Intoarce pe coloane: caroserie, inclinare, roata fata, roata spate,
    acceleratia pilotului, apasarea relativa fata si spate (1 = cat static, 0 = roata in aer)."""
    kf, kr, cf_c, cf_d, cr_c, cr_d = map(float, gene)
    df, ds = drum if drum is not None else (DRUM_FATA, DRUM_SPATE)
    z = th = zf = zr = vz = vth = vf = vr = 0.0
    las1, las2 = SARCINA / kf, SARCINA / kr      # cat din cursa e deja consumata
    urma = []
    for hf, hr in zip(df.tolist(), ds.tolist()):
        x1, v1 = zf - z - A * th, vf - vz - A * vth  # comprimarea suspensiei, pozitiva cand se strange
        x2, v2 = zr - z + B * th, vr - vz + B * vth
        f1 = kf * x1 + (cf_c if v1 > 0 else cf_d) * v1
        f2 = kr * x2 + (cr_c if v2 > 0 else cr_d) * v2
        if abs(x1 + las1) > CURSA:
            f1 += K_TAMPON * (x1 + las1 - math.copysign(CURSA, x1 + las1))
        if abs(x2 + las2) > CURSA:
            f2 += K_TAMPON * (x2 + las2 - math.copysign(CURSA, x2 + las2))
        n1 = max(0.0, N_FATA + K_PNEU * (hf - zf))   # pneul impinge, dar nu trage: roata poate sari
        n2 = max(0.0, N_SPATE + K_PNEU * (hr - zr))
        az, ath = (f1 + f2) / M, (A * f1 - B * f2) / I
        urma.append((z, th, zf, zr, az + SCAUN * ath, n1 / N_FATA, n2 / N_SPATE))
        vz += az * dt
        vth += ath * dt
        vf += (n1 - N_FATA - f1) / M_ROATA * dt
        vr += (n2 - N_SPATE - f2) / M_ROATA * dt
        z += vz * dt
        th += vth * dt
        zf += vf * dt
        zr += vr * dt
    return np.array(urma)


def score(urma):
    """(fitness, zguduirea pilotului in m/s^2, variatia relativa a apasarii rotilor)"""
    zguduire = math.sqrt((urma[:, 4] ** 2).mean())
    apasare = math.sqrt(((urma[:, 5:] - 1) ** 2).mean())
    return zguduire + PONDERE_APASARE * apasare, zguduire, apasare


def fitness(gene):
    return score(simulate(gene))[0]


def check():
    # pe drum drept masina sta nemiscata, iar dupa un drum mai inalt se aseaza la inaltimea lui
    assert np.abs(simulate(OBISNUIT, drum=(np.zeros(500), np.zeros(500)))[:, :5]).max() < 1e-9
    urma = simulate(OBISNUIT, drum=(np.full(4000, 0.05), np.full(4000, 0.05)))
    assert np.abs(urma[-1, :4] - (0.05, 0, 0.05, 0.05)).max() < 1e-3, urma[-1]
    # pneul nu trage in jos, pasul de timp e destul de mic, iar arcurile prea moi sunt penalizate
    assert simulate(LIM_JOS)[:, 5:].min() >= 0
    x = VITEZA * DT / 8 * np.arange(PASI * 8)
    fin = score(simulate(OBISNUIT, DT / 8, (road(x), road(x - A - B))))[0]
    assert abs(fitness(OBISNUIT) - fin) < 0.02 * fin
    assert fitness([6e3, 6e3, *OBISNUIT[2:]]) > fitness(OBISNUIT)

    jos, sus = np.array(LIM_JOS), np.array(LIM_SUS)
    for seed in (1, 2, 3):
        de = DifferentialEvolution(NP, fitness, 6, LIM_SUS, LIM_JOS, GENERATII, F, CR, seed)
        cel_mai_bun = de.evolve()[1]
        rng = np.random.default_rng(seed)
        aleator = min(fitness(jos + rng.random(6) * (sus - jos)) for _ in range(NP * (GENERATII + 1)))
        print(f"seed {seed}: obisnuit {fitness(OBISNUIT):.3f}, aleator {aleator:.3f}, DE {cel_mai_bun:.3f}")
        assert cel_mai_bun < aleator < fitness(OBISNUIT)


# ---------------- fereastra ----------------

W, H, HARTA = 1240, 720, 860   # HARTA = latimea zonei cu masinile; restul e panoul din dreapta
SCALA, VERT = 60, 2            # pixeli pe metru; miscarile verticale sunt desenate de VERT ori mai mari
X_MASINA = 500                 # unde sta puntea fata pe ecran
BANDA = H // 4
CER, ZAPADA, PANOU, TEXT, GRI = (203, 222, 235), (240, 248, 253), (15, 23, 42), (226, 232, 240), (148, 163, 184)
INCHIS, ROSU, VERDE, ALBASTRU, PORTOCALIU = (30, 41, 59), (239, 68, 68), (34, 197, 94), (56, 189, 248), (251, 146, 60)


class Evolution:
    """Ruleaza DE pe un fir separat (o generatie dureaza ~0,2 s), ca animatia sa ramana fluida."""

    def __init__(self, seed):
        self.seed, self.gen = seed, 0
        self.urme = {}         # gene -> simularea lor, ca sa nu o refacem pentru desen
        self.durata_gen, self.pauza, self.oprit = 0.5, False, False
        self.timpi, self.medii = [], []
        self.de = DifferentialEvolution(NP, self.evaluate, 6, LIM_SUS, LIM_JOS, GENERATII, F, CR, seed)
        self.publish()
        threading.Thread(target=self.run, daemon=True).start()

    def evaluate(self, gene):
        urma = simulate(gene)
        self.urme[gene.tobytes()] = urma
        return score(urma)[0]

    def publish(self):
        fit = np.array([c.val_fitness_function for c in self.de.population])
        gene = np.array([c.genes for c in self.de.population])
        urme = [self.urme[g.tobytes()] for g in gene]
        self.urme = {g.tobytes(): u for g, u in zip(gene, urme)}
        self.timpi.append(float(fit.min()))
        self.medii.append(float(fit.mean()))
        self.stare = (self.gen, gene, fit, urme)  # o singura atribuire: fereastra vede mereu o stare completa

    def run(self):
        while self.gen < GENERATII and not self.oprit:
            inceput = time.time()
            if self.pauza:
                time.sleep(0.05)
                continue
            self.de.selection(minimize=True)
            self.gen += 1
            self.publish()
            while time.time() - inceput < self.durata_gen and not self.oprit:
                time.sleep(0.02)


def make_lanes(evo, prima):
    """Cele 4 benzi: (titlu, simularile desenate, care dintre ele e in fata)."""
    gen, _, fit, urme = evo.stare
    best = int(fit.argmin())
    obisnuit = simulate(OBISNUIT)
    return [(f"generatia 0, individul median: fitness {score(prima)[0]:.2f}", [prima], 0),
            (f"reglaj obisnuit: fitness {score(obisnuit)[0]:.2f}", [obisnuit], 0),
            (f"cel mai bun din generatia {gen}: fitness {fit[best]:.2f}", [urme[best]], 0),
            (f"toata populatia din generatia {gen}", urme, best)]


def median_run(evo):
    _, _, fit, urme = evo.stare
    return urme[int(np.argsort(fit)[len(fit) // 2])]


def draw_penguin(ecran, x, y):
    """Pinguinul pilot, din profil, cu privirea spre dreapta; (x, y) = scaunul."""
    pygame.draw.ellipse(ecran, (20, 24, 34), (x - 13, y - 40, 26, 44))
    pygame.draw.ellipse(ecran, (245, 245, 245), (x - 2, y - 28, 14, 30))
    pygame.draw.circle(ecran, (20, 24, 34), (x + 2, y - 42), 12)
    pygame.draw.circle(ecran, (245, 245, 245), (x + 6, y - 44), 4)
    pygame.draw.circle(ecran, (20, 24, 34), (x + 7, y - 44), 2)
    pygame.draw.polygon(ecran, PORTOCALIU, [(x + 12, y - 43), (x + 22, y - 39), (x + 12, y - 36)])
    pygame.draw.line(ecran, ROSU, (x - 10, y - 31), (x + 11, y - 31), 5)
    pygame.draw.line(ecran, ROSU, (x - 10, y - 31), (x - 22, y - 26), 4)


def draw_car(ecran, urma, k, sol, fantoma=False):
    z, th, zf, zr, _, nf, nr = urma[k]
    s = SCALA * VERT
    centru = X_MASINA - A * SCALA
    y_caroserie = lambda dx: sol - 46 - s * (z + th * dx)  # noqa: E731
    colturi = [(centru + dx * SCALA, y_caroserie(dx) + dy) for dx, dy in
               ((-2.0, 0), (1.9, 0), (1.9, -14), (0.8, -20), (0.45, -34), (0.35, -20), (-0.9, -20), (-2.0, -24))]
    if fantoma:
        pygame.draw.polygon(ecran, (100, 116, 139), colturi, 1)
        return
    for dx, zw, n in ((A, zf, nf), (-B, zr, nr)):
        x, y = centru + dx * SCALA, sol - 18 - s * zw
        pygame.draw.line(ecran, PORTOCALIU, (x, y), (x, y_caroserie(dx)), 5)  # suspensia
        pygame.draw.circle(ecran, INCHIS, (x, y), 18)
        pygame.draw.circle(ecran, ROSU if n < 0.02 else GRI, (x, y), 9)       # rosu = roata e in aer
    draw_penguin(ecran, centru + SCAUN * SCALA, y_caroserie(SCAUN) - 12)
    pygame.draw.polygon(ecran, (37, 99, 235), colturi)


def draw_lane(ecran, font, mic, nr, titlu, urme, fata, k):
    sus, sol = nr * BANDA, (nr + 1) * BANDA - 34
    pygame.draw.rect(ecran, CER, (0, sus, HARTA, BANDA))
    px = np.arange(0, HARTA + 8, 4)
    y = sol - SCALA * VERT * road(X_FATA[k] + (px - X_MASINA) / SCALA)
    puncte = list(zip(px.tolist(), y.tolist()))
    pygame.draw.polygon(ecran, ZAPADA, puncte + [(HARTA + 8, sus + BANDA), (0, sus + BANDA)])
    pygame.draw.lines(ecran, (71, 85, 105), False, puncte, 3)
    if len(urme) > 1:
        for u in urme:
            draw_car(ecran, u, k, sol, fantoma=True)
    draw_car(ecran, urme[fata], k, sol)
    pygame.draw.line(ecran, INCHIS, (0, sus), (HARTA, sus), 2)
    ecran.blit(font.render(titlu, True, INCHIS), (12, sus + 8))
    # ce simte pilotul: acceleratia din ultimele 2 secunde
    x0, w, h, n = HARTA - 232, 220, 56, int(2 / DT)
    pygame.draw.rect(ecran, ZAPADA, (x0, sus + 8, w, h), border_radius=4)
    acc = urme[fata][max(0, k - n):k + 1, 4]
    if len(acc) > 1:
        ya = sus + 8 + h / 2 - np.clip(acc / 15, -1, 1) * (h / 2 - 2)
        xa = np.linspace(x0, x0 + w * len(acc) / (n + 1), len(acc))
        pygame.draw.lines(ecran, ROSU, False, list(zip(xa, ya)), 1)
    ecran.blit(mic.render(f"zguduire pe tot drumul: {score(urme[fata])[1]:.2f} m/s2", True, INCHIS), (x0, sus + 68))


def draw_genes(ecran, mic, gene, best, x, y, w, h):
    """Coordonate paralele: o linie = un individ, o axa = o gena. Liniile se strang cand populatia converge."""
    pygame.draw.rect(ecran, INCHIS, (x, y, w, h), border_radius=6)
    axe = np.linspace(x + 26, x + w - 26, 6)
    rel = (gene - LIM_JOS) / (np.array(LIM_SUS) - LIM_JOS)
    yy = y + h - 34 - rel * (h - 50)
    for a in axe:
        pygame.draw.line(ecran, (71, 85, 105), (a, y + 14), (a, y + h - 34))
    for i, linie in enumerate(yy):
        pygame.draw.aalines(ecran, (120, 140, 170), False, list(zip(axe, linie)))
    pygame.draw.lines(ecran, VERDE, False, list(zip(axe, yy[best])), 3)
    for k, (a, nume) in enumerate(zip(axe, NUME)):
        text = mic.render(nume, True, GRI)
        jumatate = text.get_width() / 2 + 2
        centru = min(max(a, x + jumatate), x + w - jumatate)
        ecran.blit(text, text.get_rect(midtop=(centru, y + h - 30 + 14 * (k % 2))))


def draw_chart(ecran, mic, evo, x, y, w, h):
    pygame.draw.rect(ecran, INCHIS, (x, y, w, h), border_radius=6)
    timpi, medii = list(evo.timpi), list(evo.medii)
    lo, hi = min(timpi) - 0.2, max(medii) + 0.2
    for valori, culoare in ((medii, PORTOCALIU), (timpi, ALBASTRU)):
        if len(valori) > 1:
            pygame.draw.lines(ecran, culoare, False, [
                (x + 8 + k / GENERATII * (w - 16), y + h - 8 - (v - lo) / (hi - lo) * (h - 16))
                for k, v in enumerate(valori)], 2)
    for k, (nume, culoare) in enumerate((("cel mai bun", ALBASTRU), ("media populatiei", PORTOCALIU))):
        ecran.blit(mic.render(nume, True, culoare), (x + w - 130, y + 8 + 16 * k))


def main():
    pygame.init()
    ecran = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Suspensia pinguinului: evolutie diferentiala")
    mare, font, mic = (pygame.font.Font(None, m) for m in (34, 24, 18))
    ceas = pygame.time.Clock()

    evo = Evolution(SEED)
    prima, t = median_run(evo), 0.0
    benzi, gene_desen = make_lanes(evo, prima), evo.stare[1].copy()

    while True:
        dt = min(ceas.tick(60) / 1000, 0.05)
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                evo.oprit = True
                pygame.quit()
                return
            if e.type != pygame.KEYDOWN:
                continue
            if e.key == pygame.K_SPACE:
                evo.pauza = not evo.pauza
            elif e.key == pygame.K_r:
                evo.oprit = True
                evo = Evolution(evo.seed + 1)
                prima, t = median_run(evo), 0.0
                benzi, gene_desen = make_lanes(evo, prima), evo.stare[1].copy()
            elif e.key == pygame.K_UP:
                evo.durata_gen = max(0.1, evo.durata_gen / 2)
            elif e.key == pygame.K_DOWN:
                evo.durata_gen = min(4.0, evo.durata_gen * 2)

        if not evo.pauza:
            t += dt
            if t >= PASI * DT:  # drumul s-a terminat: il luam de la capat, cu cele mai noi masini
                t = 0.0
                benzi = make_lanes(evo, prima)
        k = min(int(t / DT), PASI - 1)
        gen, gene, fit, _ = evo.stare
        gene_desen += 0.15 * (gene - gene_desen)  # populatia se muta lin de la o generatie la alta
        best = int(fit.argmin())

        ecran.fill(PANOU)
        for nr, (titlu, urme, fata) in enumerate(benzi):
            draw_lane(ecran, font, mic, nr, titlu, urme, fata, k)
        pygame.draw.rect(ecran, PANOU, (HARTA, 0, W - HARTA, H))
        x, w = HARTA + 20, W - HARTA - 40
        ecran.blit(mare.render(f"Generatia {gen} / {GENERATII}", True, TEXT), (x, 18))
        for i, linie in enumerate((f"cel mai bun fitness: {fit.min():.3f}", f"media populatiei: {fit.mean():.3f}",
                                   f"NP {NP}, gene 6, F {F}, CR {CR}, seed {evo.seed}")):
            ecran.blit((font if i < 2 else mic).render(linie, True, GRI), (x, 54 + 24 * i))
        ecran.blit(mic.render("genele populatiei (verde = cel mai bun)", True, GRI), (x, 136))
        draw_genes(ecran, mic, gene_desen, best, x, 154, w, 220)
        ecran.blit(mic.render("fitness pe generatii (mai mic = mai bine)", True, GRI), (x, 390))
        draw_chart(ecran, mic, evo, x, 408, w, 150)
        for i, linie in enumerate((f"miscarile verticale sunt desenate de {VERT} ori mai mari",
                                   "roata rosie = roata e in aer",
                                   "SPATIU pauza" + (" (activa)" if evo.pauza else ""), "R alt seed",
                                   "SUS / JOS viteza evolutiei", "ESC iesire")):
            ecran.blit(mic.render(linie, True, GRI), (x, 576 + 22 * i))
        pygame.display.flip()


if __name__ == "__main__":
    check() if "--check" in sys.argv else main()
