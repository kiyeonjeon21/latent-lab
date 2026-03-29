"""FastAPI model serving endpoint."""

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Latent Lab Model Server")


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7
    model: str = ""


class GenerateResponse(BaseModel):
    text: str
    tokens_generated: int
    model: str


# Model cache
_models: dict = {}


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generate text from a loaded model."""
    from latent_lab.models.mlx_utils import generate as mlx_generate
    from latent_lab.models.mlx_utils import load_mlx_model

    model_key = request.model
    if model_key not in _models:
        _models[model_key] = load_mlx_model(model_key)

    model, tokenizer = _models[model_key]
    result = mlx_generate(
        model,
        tokenizer,
        prompt=request.prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
    )
    return GenerateResponse(
        text=result,
        tokens_generated=len(result.split()),
        model=model_key,
    )


@app.get("/health")
async def health():
    return {"status": "ok", "models_loaded": list(_models.keys())}
