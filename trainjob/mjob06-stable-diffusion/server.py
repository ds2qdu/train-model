# ============================================
# Stable Diffusion - Image Generation Server
# FastAPI 기반 이미지 생성 API
# ============================================

import os
import io
import base64
import torch
from pathlib import Path
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="Stable Diffusion Image Generation API")

# Global variables for model
pipe = None
lora_loaded = False

class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = "blurry, bad quality, distorted"
    num_inference_steps: Optional[int] = 50
    guidance_scale: Optional[float] = 7.5
    width: Optional[int] = 512
    height: Optional[int] = 512
    seed: Optional[int] = None
    use_lora: Optional[bool] = True

class GenerateResponse(BaseModel):
    image_base64: str
    prompt: str
    seed: int
    generation_time: float

def load_model():
    """Load Stable Diffusion model"""
    global pipe, lora_loaded

    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

    print("Loading Stable Diffusion model...")

    model_path = os.environ.get("SD_MODEL_PATH", "runwayml/stable-diffusion-v1-5")
    lora_path = Path("/mnt/storage/models/lora_final")

    # Load base model
    pipe = StableDiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        safety_checker=None,
        cache_dir=os.environ.get('HF_HOME', None)
    )

    # Use faster scheduler
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    # Move to GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)

    # Enable memory optimization
    if hasattr(pipe, 'enable_xformers_memory_efficient_attention'):
        try:
            pipe.enable_xformers_memory_efficient_attention()
            print("xformers memory efficient attention enabled")
        except:
            pass

    # Load LoRA weights if available
    if lora_path.exists():
        try:
            from peft import PeftModel
            print(f"Loading LoRA weights from {lora_path}")
            pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_path)
            lora_loaded = True
            print("LoRA weights loaded successfully")
        except Exception as e:
            print(f"Failed to load LoRA weights: {e}")
            lora_loaded = False
    else:
        print("No LoRA weights found, using base model")
        lora_loaded = False

    print(f"Model loaded on {device}")
    return pipe

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()

@app.get("/")
async def root():
    """Health check"""
    return {
        "status": "running",
        "model": "Stable Diffusion",
        "lora_loaded": lora_loaded,
        "gpu": torch.cuda.is_available()
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "lora_loaded": lora_loaded}

@app.post("/generate", response_model=GenerateResponse)
async def generate_image(request: GenerateRequest):
    """Generate image from text prompt"""
    global pipe

    if pipe is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        start_time = datetime.now()

        # Set seed for reproducibility
        if request.seed is not None:
            generator = torch.Generator(device="cuda").manual_seed(request.seed)
            seed = request.seed
        else:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
            generator = torch.Generator(device="cuda").manual_seed(seed)

        # Generate image
        with torch.autocast("cuda"):
            result = pipe(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                num_inference_steps=request.num_inference_steps,
                guidance_scale=request.guidance_scale,
                width=request.width,
                height=request.height,
                generator=generator
            )

        image = result.images[0]

        # Convert to base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        # Save to disk
        output_dir = Path("/mnt/storage/generated")
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = output_dir / f"generated_{timestamp}_{seed}.png"
        image.save(image_path)

        generation_time = (datetime.now() - start_time).total_seconds()

        return GenerateResponse(
            image_base64=img_base64,
            prompt=request.prompt,
            seed=seed,
            generation_time=generation_time
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def list_models():
    """List available models and LoRA weights"""
    models_dir = Path("/mnt/storage/models")
    lora_models = []

    if models_dir.exists():
        for item in models_dir.iterdir():
            if item.is_dir() and (item / "adapter_config.json").exists():
                lora_models.append(item.name)

    return {
        "base_model": "runwayml/stable-diffusion-v1-5",
        "lora_models": lora_models,
        "current_lora_loaded": lora_loaded
    }

@app.post("/reload")
async def reload_model():
    """Reload model (useful after training new LoRA)"""
    global pipe, lora_loaded
    pipe = None
    load_model()
    return {"status": "reloaded", "lora_loaded": lora_loaded}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
