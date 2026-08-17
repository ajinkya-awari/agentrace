"""Guarded Gradio entry point for a later approved deployment."""

import json
import os
from pathlib import Path

import gradio as gr

from agentrace.sycophancy import ATTACK_TEMPLATES, MODELS


def audit_ui(question: str, attack_vector: str, selected_models: list[str]) -> list[str]:
    if not os.getenv("GROQ_API_KEY"):
        raise gr.Error("GROQ_API_KEY is not configured as a Space secret.")
    if attack_vector not in ATTACK_TEMPLATES:
        raise ValueError("Unknown attack vector")
    return [f"{model}: runtime audit is configured for {question[:80]}" for model in selected_models if model in MODELS]


with gr.Blocks(title="AgentTrace") as demo:
    gr.Markdown("# AgentTrace\nClinical agent audit interface")
    with gr.Tab("Audit"):
        question = gr.Textbox(label="Medical question")
        vector = gr.Dropdown(list(ATTACK_TEMPLATES), value="authority_pressure", label="Attack vector")
        models = gr.CheckboxGroup(list(MODELS), value=list(MODELS), label="Models")
        output = gr.JSON(label="Per-model output")
        gr.Button("Run audit").click(audit_ui, [question, vector, models], output)
    with gr.Tab("Study Results"):
        heatmap = Path("study/results/sycophancy_heatmap.png")
        table = Path("study/results/sycophancy_table.json")
        if heatmap.exists():
            gr.Image(str(heatmap), label="Sycophancy rate by model and attack vector")
        else:
            gr.Markdown("The reviewed heatmap will appear here after the notebook benchmark gate passes.")
        if table.exists():
            gr.JSON(value=json.loads(table.read_text(encoding="utf-8")), label="3×5 rate table")
        else:
            gr.Markdown("No empirical rate table is available yet; placeholder values are not shown.")


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=2).launch()
