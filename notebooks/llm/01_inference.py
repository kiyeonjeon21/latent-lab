"""MLX LLM inference playground."""
# /// script
# requires-python = ">=3.12"
# dependencies = ["marimo", "mlx-lm"]
# ///

import marimo

__generated_with = "0.9.0"
app = marimo.App(width="medium")


@app.cell
def header():
    import marimo as mo

    mo.md("# MLX LLM Inference Playground")
    return (mo,)


@app.cell
def model_selector():
    import marimo as mo

    model_id = mo.ui.dropdown(
        options=[
            "mlx-community/Llama-3.2-3B-Instruct-4bit",
            "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
            "mlx-community/Qwen2.5-7B-Instruct-4bit",
            "mlx-community/Qwen2.5-14B-Instruct-4bit",
        ],
        value="mlx-community/Llama-3.2-3B-Instruct-4bit",
        label="Model",
    )
    mo.md(f"## Select Model\n{model_id}")
    return (model_id,)


@app.cell
def parameters():
    import marimo as mo

    max_tokens = mo.ui.slider(32, 1024, value=256, step=32, label="Max tokens")
    temperature = mo.ui.slider(0.0, 2.0, value=0.7, step=0.1, label="Temperature")
    top_p = mo.ui.slider(0.0, 1.0, value=0.9, step=0.05, label="Top-p")

    mo.md(f"## Parameters\n{max_tokens}\n{temperature}\n{top_p}")
    return max_tokens, temperature, top_p


@app.cell
def inference(model_id, max_tokens, temperature, top_p):
    import marimo as mo

    prompt = mo.ui.text_area(
        value="Explain the concept of attention in transformers in simple terms.",
        label="Prompt",
        full_width=True,
    )
    run_btn = mo.ui.run_button(label="Generate")
    mo.md(f"{prompt}\n\n{run_btn}")

    if run_btn.value:
        from mlx_lm import generate, load

        mo.md("Loading model...")
        model, tokenizer = load(model_id.value)

        mo.md("Generating...")
        result = generate(
            model,
            tokenizer,
            prompt=prompt.value,
            max_tokens=max_tokens.value,
            temp=temperature.value,
            top_p=top_p.value,
        )
        mo.md(f"### Response\n\n{result}")

    return


if __name__ == "__main__":
    app.run()
