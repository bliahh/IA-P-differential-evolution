"""Interfata proiectului: cate un tab pentru fiecare coleg. Rulare: python app.py"""
import gradio as gr

COLEGI = ["Bliah", "Avornicesei", "Spiridon"]

with gr.Blocks(title="Evolutie diferentiala") as demo:
    gr.Markdown("# Evolutie diferentiala: aplicatii")
    for coleg in COLEGI:
        with gr.Tab(coleg):
            gr.Markdown("Aplicatia e in lucru.")

if __name__ == "__main__":
    demo.launch()
