"""Interfata proiectului: cate un tab pentru fiecare coleg. Rulare: python app.py"""
import subprocess
import sys
from pathlib import Path

import gradio as gr

cursa = None  # fereastra pygame pornita din interfata


def open_race():
    """Porneste cursa intr-o fereastra separata (merge doar pe calculatorul pe care ruleaza interfata)."""
    global cursa
    if cursa and cursa.poll() is None:
        return "Cursa e deja deschisa."
    cursa = subprocess.Popen([sys.executable, str(Path(__file__).parent / "penguin_race.py")])
    try:
        cursa.wait(timeout=2)  # daca se inchide imediat, inseamna ca nu a putut porni
    except subprocess.TimeoutExpired:
        return "Am deschis cursa intr-o fereastra separata (ESC o inchide)."
    return "Cursa nu a pornit. Vezi eroarea in terminalul in care ruleaza interfata."


with gr.Blocks(title="Evolutie diferentiala") as demo:
    gr.Markdown("# Evolutie diferentiala: aplicatii")
    with gr.Tab("Bliah"):
        gr.Markdown("Aplicatia e in lucru.")
    with gr.Tab("Avornicesei"):
        gr.Markdown("### Cursa pinguinilor\nEvolutia diferentiala cauta cea mai rapida trasa pe un circuit de gheata. "
                    "Un individ e o trasa intreaga, iar fitness-ul e timpul pe tur.")
        buton = gr.Button("Deschide cursa (pygame)", variant="primary")
        buton.click(open_race, outputs=gr.Markdown())
    with gr.Tab("Spiridon"):
        gr.Markdown("Aplicatia e in lucru.")

if __name__ == "__main__":
    demo.launch()
