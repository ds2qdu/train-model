# ============================================
# Stable Diffusion - Chatbot Backend
# LLM + Image Generation Integration
# ============================================

import os
import asyncio
import httpx
from typing import Optional, List
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="Stable Diffusion Chatbot API")

# Configuration
SD_SERVER_URL = os.environ.get("SD_SERVER_URL", "http://stable-diffusion-server:8000")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

# System prompts
SYSTEM_PROMPT = """You are an AI assistant specialized in image generation using Stable Diffusion.
Your role is to help users create amazing images by:
1. Understanding their creative vision
2. Crafting effective prompts for Stable Diffusion
3. Explaining image generation concepts

When a user wants to generate an image:
- Help them refine their prompt for better results
- Suggest improvements like style, lighting, composition
- Add quality-enhancing keywords like: "highly detailed, 8k, professional photography, cinematic lighting"

For negative prompts, suggest: "blurry, bad quality, distorted, deformed, ugly, low resolution"

You can understand Korean and respond in the same language as the user."""

class ChatRequest(BaseModel):
    message: str
    generate_image: Optional[bool] = False
    image_settings: Optional[dict] = None

class ChatResponse(BaseModel):
    response: str
    image_base64: Optional[str] = None
    prompt_used: Optional[str] = None
    seed: Optional[int] = None

class PromptOptimizeRequest(BaseModel):
    user_prompt: str
    style: Optional[str] = None

def is_image_request(message: str) -> bool:
    """Check if message is requesting image generation"""
    image_keywords = [
        'generate', 'create', 'make', 'draw', 'paint',
        '생성', '만들', '그려', '그림', '이미지',
        'image', 'picture', 'photo', 'art', 'illustration'
    ]
    message_lower = message.lower()
    return any(keyword in message_lower for keyword in image_keywords)

def extract_prompt_from_message(message: str) -> str:
    """Extract image prompt from user message"""
    # Remove common request phrases
    remove_phrases = [
        'generate an image of', 'create an image of', 'make an image of',
        'draw me', 'paint me', 'generate', 'create', 'make',
        '이미지 생성해줘', '그림 그려줘', '만들어줘', '생성해줘',
        'please', '해줘', '줘'
    ]

    prompt = message.lower()
    for phrase in remove_phrases:
        prompt = prompt.replace(phrase, '')

    return prompt.strip()

async def call_ollama(prompt: str, system_prompt: str = SYSTEM_PROMPT) -> str:
    """Call Ollama LLM"""
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": "llama3.2:1b",
                    "prompt": prompt,
                    "system": system_prompt,
                    "stream": False
                }
            )
            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                return f"LLM Error: {response.status_code}"
    except Exception as e:
        return f"LLM connection error: {str(e)}"

async def optimize_prompt(user_prompt: str, style: str = None) -> str:
    """Use LLM to optimize the image prompt"""
    optimization_prompt = f"""Convert this user request into an optimized Stable Diffusion prompt.

User request: {user_prompt}
{"Style preference: " + style if style else ""}

Create a detailed prompt that includes:
1. Subject description
2. Art style (if not specified, suggest one)
3. Lighting and atmosphere
4. Quality keywords

Return ONLY the optimized prompt, nothing else. Keep it concise but descriptive."""

    optimized = await call_ollama(optimization_prompt)

    # Add quality enhancers if not present
    quality_keywords = ["highly detailed", "8k", "professional"]
    if not any(kw in optimized.lower() for kw in quality_keywords):
        optimized += ", highly detailed, 8k resolution, professional quality"

    return optimized.strip()

async def generate_image(prompt: str, settings: dict = None) -> dict:
    """Call Stable Diffusion server to generate image"""
    try:
        request_data = {
            "prompt": prompt,
            "negative_prompt": "blurry, bad quality, distorted, deformed, ugly, low resolution, amateur",
            "num_inference_steps": 30,
            "guidance_scale": 7.5,
            "width": 512,
            "height": 512
        }

        if settings:
            request_data.update(settings)

        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"{SD_SERVER_URL}/generate",
                json=request_data
            )

            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"SD Server Error: {response.status_code}"}

    except Exception as e:
        return {"error": f"SD Server connection error: {str(e)}"}

@app.get("/")
async def root():
    """Health check"""
    return {"status": "running", "service": "SD Chatbot"}

@app.get("/health")
async def health():
    """Health check with service status"""
    sd_status = "unknown"
    ollama_status = "unknown"

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{SD_SERVER_URL}/health")
            sd_status = "healthy" if resp.status_code == 200 else "unhealthy"
    except:
        sd_status = "unreachable"

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{OLLAMA_URL}/api/tags")
            ollama_status = "healthy" if resp.status_code == 200 else "unhealthy"
    except:
        ollama_status = "unreachable"

    return {
        "status": "running",
        "sd_server": sd_status,
        "ollama": ollama_status
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint with optional image generation"""

    message = request.message
    should_generate = request.generate_image or is_image_request(message)

    if should_generate:
        # Extract and optimize prompt
        raw_prompt = extract_prompt_from_message(message)
        optimized_prompt = await optimize_prompt(raw_prompt)

        # Generate response explaining what we're doing
        response_text = f"이미지를 생성하고 있습니다...\n\n**사용 프롬프트:**\n{optimized_prompt}"

        # Generate image
        result = await generate_image(optimized_prompt, request.image_settings)

        if "error" in result:
            return ChatResponse(
                response=f"이미지 생성 중 오류가 발생했습니다: {result['error']}",
                image_base64=None,
                prompt_used=optimized_prompt
            )

        return ChatResponse(
            response=f"이미지가 생성되었습니다!\n\n**사용된 프롬프트:** {optimized_prompt}\n**시드:** {result.get('seed')}\n**생성 시간:** {result.get('generation_time', 0):.1f}초",
            image_base64=result.get("image_base64"),
            prompt_used=optimized_prompt,
            seed=result.get("seed")
        )

    else:
        # Regular chat response
        llm_response = await call_ollama(message)
        return ChatResponse(response=llm_response)

@app.post("/optimize-prompt")
async def optimize_prompt_endpoint(request: PromptOptimizeRequest):
    """Optimize a prompt for better image generation"""
    optimized = await optimize_prompt(request.user_prompt, request.style)
    return {"original": request.user_prompt, "optimized": optimized}

@app.post("/generate-direct")
async def generate_direct(request: ChatRequest):
    """Generate image directly without LLM optimization"""
    result = await generate_image(request.message, request.image_settings)

    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])

    return result

@app.get("/styles")
async def get_styles():
    """Get available style presets"""
    return {
        "styles": [
            {"name": "realistic", "prompt_suffix": ", photorealistic, 8k, professional photography"},
            {"name": "anime", "prompt_suffix": ", anime style, vibrant colors, detailed"},
            {"name": "oil_painting", "prompt_suffix": ", oil painting, classical art style, rich textures"},
            {"name": "watercolor", "prompt_suffix": ", watercolor painting, soft colors, artistic"},
            {"name": "cyberpunk", "prompt_suffix": ", cyberpunk style, neon lights, futuristic"},
            {"name": "fantasy", "prompt_suffix": ", fantasy art, magical, epic composition"},
            {"name": "minimalist", "prompt_suffix": ", minimalist design, clean lines, simple"},
            {"name": "3d_render", "prompt_suffix": ", 3D render, octane render, highly detailed"}
        ]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
