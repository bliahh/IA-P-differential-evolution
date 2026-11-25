"""Interfata proiectului: cate un tab pentru fiecare coleg. Rulare: python app.py"""
import subprocess
import sys
from pathlib import Path

import gradio as gr

fereastra = None  # fereastra pygame pornita din interfata


def open_suspension():
    """Porneste aplicatia intr-o fereastra separata (merge doar pe calculatorul pe care ruleaza interfata)."""
    global fereastra
    if fereastra and fereastra.poll() is None:
        return "Aplicatia e deja deschisa."
    fereastra = subprocess.Popen([sys.executable, str(Path(__file__).parent / "penguin_suspension.py")])
    try:
        fereastra.wait(timeout=2)  # daca se inchide imediat, inseamna ca nu a putut porni
    except subprocess.TimeoutExpired:
        return "Am deschis aplicatia intr-o fereastra separata (ESC o inchide)."
    return "Aplicatia nu a pornit. Vezi eroarea in terminalul in care ruleaza interfata."


with gr.Blocks(title="Evolutie diferentiala") as demo:
    gr.Markdown("# Evolutie diferentiala: aplicatii")
    with gr.Tab("Bliah"):
        gr.Markdown("Aplicatia e in lucru.")
    with gr.Tab("Avornicesei"):
        gr.Markdown("### Suspensia pinguinului\nEvolutia diferentiala regleaza arcurile si amortizoarele unei masini "
                    "cu pilot pinguin, pe un drum cu denivelari. Un individ e un reglaj complet (6 gene), iar "
                    "fitness-ul e cat e zguduit pilotul plus cat variaza apasarea rotilor pe drum.")
        buton = gr.Button("Deschide suspensia (pygame)", variant="primary")
        buton.click(open_suspension, outputs=gr.Markdown())
    with gr.Tab("Spiridon"):
        gr.Markdown("Aplicatia e in lucru.")

if __name__ == "__main__":
    demo.launch()
