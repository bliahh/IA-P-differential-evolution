"""Interfața proiectului: câte un tab pentru fiecare problemă. Rulare: python app.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "common"))  # algoritmul face `from Chromosome import ...`

import gradio as gr  # noqa: E402

from problems import pinguini  # noqa: E402

# Fiecare coleg își adaugă aici problema: (titlul tabului, funcția care construiește conținutul tabului).
TABURI = [
    ("Pinguini pe rampă", pinguini.build_tab),
]

DESPRE = """
## Evoluție diferențială (DE/rand/1/bin)
Fiecare tab de mai sus/jos prezintă o problemă de optimizare cu reprezentare reală, rezolvată cu
același algoritm (`common/DifferentialEvolution.py`): mutație `v = a + F·(b − c)`, încrucișare binomială
cu rata CR și selecție lacomă 1 la 1.
"""

with gr.Blocks(title="Evoluție diferențială") as demo:
    gr.Markdown("# Evoluție diferențială: aplicații")
    for titlu, build_tab in TABURI:
        with gr.Tab(titlu):
            build_tab()
    with gr.Tab("Despre"):
        gr.Markdown(DESPRE)

if __name__ == "__main__":
    demo.launch()
