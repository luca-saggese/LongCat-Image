"""
LongCat-Image OpenAI-Compatible API Server

This server exposes text-to-image generation and image editing capabilities
through OpenAI-compatible endpoints.
"""

import base64
import io
import os
import torch
import subprocess
import gc
from typing import Optional, List
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import AutoProcessor
from PIL import Image

from longcat_image.models import LongCatImageTransformer2DModel
from longcat_image.pipelines import LongCatImagePipeline, LongCatImageEditPipeline

# Memory optimization
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ==================== Configuration ====================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
T2I_CHECKPOINT = os.getenv('T2I_CHECKPOINT', './weights/LongCat-Image')
EDIT_CHECKPOINT = os.getenv('EDIT_CHECKPOINT', './weights/LongCat-Image-Edit')
USE_CPU_OFFLOAD = os.getenv('USE_CPU_OFFLOAD', 'true').lower() == 'true'
MAX_BATCH_SIZE = int(os.getenv('MAX_BATCH_SIZE', '1'))

# Global pipeline instances
t2i_pipe = None
edit_pipe = None
t2i_loaded = False
edit_loaded = False


# ==================== Models ====================
class ImageResponseFormat(BaseModel):
    """Response format for generated images"""
    type: str = "b64_json"  # or "url"


class TextToImageRequest(BaseModel):
    """OpenAI-compatible text-to-image request"""
    prompt: str
    negative_prompt: Optional[str] = ""
    n: int = 1  # number of images
    size: Optional[str] = "1344x768"  # width x height
    guidance_scale: Optional[float] = 4.5
    num_inference_steps: Optional[int] = 30  # Reduced from 50 for memory
    seed: Optional[int] = None
    response_format: Optional[str] = "b64_json"


class ImageEditRequest(BaseModel):
    """OpenAI-compatible image edit request"""
    prompt: str
    negative_prompt: Optional[str] = ""
    n: int = 1
    guidance_scale: Optional[float] = 4.5
    num_inference_steps: Optional[int] = 30  # Reduced from 50 for memory
    seed: Optional[int] = None
    response_format: Optional[str] = "b64_json"


class ImageResponse(BaseModel):
    """Response containing generated images"""
    created: int
    data: List[dict]


# ==================== Initialization ====================
def download_model(model_name: str, checkpoint_dir: str):
    """Download model from HuggingFace if not present"""
    print(f"Downloading {model_name} model...")
    
    try:
        subprocess.run([
            "huggingface-cli", 
            "download", 
            model_name,
            "--local-dir",
            checkpoint_dir
        ], check=True)
        print(f"✓ {model_name} downloaded successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to download {model_name}: {str(e)}")
        return False
    except FileNotFoundError:
        print(f"✗ huggingface-cli not found. Install with: pip install huggingface-hub")
        return False


def check_and_download_models():
    """Check if models exist, download if missing"""
    print("\n" + "="*60)
    print("Checking models...")
    print("="*60)
    
    # Check T2I model
    t2i_model_exists = Path(T2I_CHECKPOINT).exists() and any(
        Path(T2I_CHECKPOINT).glob("**/*transformer*")
    )
    
    if not t2i_model_exists:
        print(f"\n⚠  T2I model not found at {T2I_CHECKPOINT}")
        if not download_model("meituan-longcat/LongCat-Image", T2I_CHECKPOINT):
            print(f"⚠  Could not download T2I model")
            print(f"   Download manually: huggingface-cli download meituan-longcat/LongCat-Image --local-dir {T2I_CHECKPOINT}")
    else:
        print(f"✓ T2I model found at {T2I_CHECKPOINT}")
    
    # Check Edit model
    edit_model_exists = Path(EDIT_CHECKPOINT).exists() and any(
        Path(EDIT_CHECKPOINT).glob("**/*transformer*")
    )
    
    if not edit_model_exists:
        print(f"\n⚠  Edit model not found at {EDIT_CHECKPOINT}")
        if not download_model("meituan-longcat/LongCat-Image-Edit", EDIT_CHECKPOINT):
            print(f"⚠  Could not download Edit model")
            print(f"   Download manually: huggingface-cli download meituan-longcat/LongCat-Image-Edit --local-dir {EDIT_CHECKPOINT}")
    else:
        print(f"✓ Edit model found at {EDIT_CHECKPOINT}")
    
    print("="*60 + "\n")


def unload_all_pipelines():
    """Unload all pipelines to free memory"""
    global t2i_pipe, edit_pipe, t2i_loaded, edit_loaded
    
    print("Unloading pipelines to free memory...")
    
    if t2i_pipe is not None:
        del t2i_pipe
        t2i_pipe = None
        t2i_loaded = False
    
    if edit_pipe is not None:
        del edit_pipe
        edit_pipe = None
        edit_loaded = False
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("✓ Memory freed")


def load_t2i_pipeline():
    """Load text-to-image pipeline"""
    global t2i_pipe, t2i_loaded
    
    if t2i_loaded and t2i_pipe is not None:
        print("T2I pipeline already loaded")
        return
    
    print(f"\n{'='*60}")
    print("Loading T2I pipeline...")
    print(f"{'='*60}")
    
    try:
        # Aggressive memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        gc.collect()
        
        text_processor = AutoProcessor.from_pretrained(
            T2I_CHECKPOINT, 
            subfolder='tokenizer',
            trust_remote_code=True
        )
        print("✓ Text processor loaded")
        
        # Load transformer on CPU only
        print("Loading transformer (CPU)...")
        transformer = LongCatImageTransformer2DModel.from_pretrained(
            T2I_CHECKPOINT,
            subfolder='transformer',
            torch_dtype=torch.float16,
            device_map="cpu",
            use_safetensors=True,
            trust_remote_code=True
        )
        print("✓ Transformer loaded on CPU")
        
        # Create pipeline
        print("Creating pipeline...")
        t2i_pipe = LongCatImagePipeline.from_pretrained(
            T2I_CHECKPOINT,
            transformer=transformer,
            text_processor=text_processor,
            torch_dtype=torch.float16
        )
        print("✓ Pipeline created")
        
        # Enable optimizations
        t2i_pipe.enable_model_cpu_offload()
        t2i_pipe.enable_attention_slicing()
        t2i_pipe.enable_vae_tiling()
        
        # Additional memory optimization
        if hasattr(t2i_pipe, 'vae'):
            t2i_pipe.vae.enable_slicing()
        
        print("✓ Memory optimizations enabled")
        
        # Final cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        t2i_loaded = True
        print(f"{'='*60}")
        print("✓ T2I pipeline ready")
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"✗ Error loading T2I pipeline: {str(e)}")
        raise


def load_edit_pipeline():
    """Load image editing pipeline"""
    global edit_pipe, edit_loaded
    
    if edit_loaded and edit_pipe is not None:
        print("Edit pipeline already loaded")
        return
    
    print(f"\n{'='*60}")
    print("Loading Edit pipeline...")
    print(f"{'='*60}")
    
    try:
        # Aggressive memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        gc.collect()
        
        text_processor = AutoProcessor.from_pretrained(
            EDIT_CHECKPOINT,
            subfolder='tokenizer',
            trust_remote_code=True
        )
        print("✓ Text processor loaded")
        
        # Load transformer on CPU only
        print("Loading transformer (CPU)...")
        transformer = LongCatImageTransformer2DModel.from_pretrained(
            EDIT_CHECKPOINT,
            subfolder='transformer',
            torch_dtype=torch.float16,
            device_map="cpu",
            use_safetensors=True,
            trust_remote_code=True
        )
        print("✓ Transformer loaded on CPU")
        
        # Create pipeline
        print("Creating pipeline...")
        edit_pipe = LongCatImageEditPipeline.from_pretrained(
            EDIT_CHECKPOINT,
            transformer=transformer,
            text_processor=text_processor,
            torch_dtype=torch.float16
        )
        print("✓ Pipeline created")
        
        # Enable optimizations
        edit_pipe.enable_model_cpu_offload()
        edit_pipe.enable_attention_slicing()
        edit_pipe.enable_vae_tiling()
        
        # Additional memory optimization
        if hasattr(edit_pipe, 'vae'):
            edit_pipe.vae.enable_slicing()
        
        print("✓ Memory optimizations enabled")
        
        # Final cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        edit_loaded = True
        print(f"{'='*60}")
        print("✓ Edit pipeline ready")
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"✗ Error loading Edit pipeline: {str(e)}")
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    # Check and download models if needed
    check_and_download_models()
    
    # DO NOT load pipelines on startup - use lazy loading instead
    # This prevents OOM errors when multiple models need to load
    print("\n" + "="*60)
    print("API Server initialized (lazy loading enabled)")
    print("Pipelines will be loaded on first use")
    print("="*60 + "\n")
    
    yield
    
    # Cleanup on shutdown
    print("\nShutting down...")


# ==================== FastAPI App ====================
app = FastAPI(
    title="LongCat-Image API",
    description="OpenAI-compatible API for LongCat-Image",
    version="1.0.0",
    lifespan=lifespan
)


# ==================== Utility Functions ====================
def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str


def parse_size(size_str: str) -> tuple:
    """Parse size string (e.g., '1344x768') to (width, height)"""
    try:
        parts = size_str.split('x')
        return int(parts[0]), int(parts[1])
    except:
        return 1344, 768


# ==================== Endpoints ====================
@app.get("/v1/models")
async def list_models():
    """List available models"""
    return {
        "object": "list",
        "data": [
            {
                "id": "longcat-image-t2i",
                "object": "model",
                "owned_by": "meituan-longcat",
                "permission": [],
                "created": 1700000000,
                "root": "longcat-image-t2i",
                "parent": None
            },
            {
                "id": "longcat-image-edit",
                "object": "model",
                "owned_by": "meituan-longcat",
                "permission": [],
                "created": 1700000000,
                "root": "longcat-image-edit",
                "parent": None
            }
        ]
    }


@app.post("/v1/images/generations")
async def text_to_image(request: TextToImageRequest):
    """
    OpenAI-compatible text-to-image endpoint
    
    Compatible with: POST /v1/images/generations
    """
    try:
        # Lazy load pipeline
        if not t2i_loaded:
            print("First request - loading T2I pipeline...")
            load_t2i_pipeline()
        
        if not t2i_pipe:
            raise HTTPException(status_code=500, detail="T2I pipeline not loaded")
        
        if request.n > MAX_BATCH_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"n cannot exceed {MAX_BATCH_SIZE}"
            )
        
        # Clear memory before generation
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        width, height = parse_size(request.size or "1344x768")
        
        # Create generator if seed is provided
        generator = None
        if request.seed is not None:
            generator = torch.Generator("cpu").manual_seed(request.seed)
        
        print(f"Generating image: {request.prompt[:50]}...")
        
        # Generate images
        with torch.inference_mode(), torch.cuda.device(DEVICE):
            output = t2i_pipe(
                request.prompt,
                negative_prompt=request.negative_prompt or "",
                height=height,
                width=width,
                guidance_scale=request.guidance_scale,
                num_inference_steps=request.num_inference_steps,
                num_images_per_prompt=request.n,
                generator=generator,
                enable_cfg_renorm=False,
                enable_prompt_rewrite=False  # Disable to save memory
            )
        
        # Clear memory after generation
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Prepare response
        images_data = []
        for idx, image in enumerate(output.images):
            if request.response_format == "b64_json":
                images_data.append({
                    "b64_json": image_to_base64(image),
                    "index": idx
                })
            else:
                images_data.append({
                    "url": f"data:image/png;base64,{image_to_base64(image)}",
                    "index": idx
                })
        
        print("✓ Image generation completed")
        
        return {
            "created": int(torch.cuda.Event(enable_timing=True).record()),
            "data": images_data
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in text_to_image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/images/edits")
async def image_edit(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    negative_prompt: Optional[str] = Form(""),
    n: Optional[int] = Form(1),
    guidance_scale: Optional[float] = Form(4.5),
    num_inference_steps: Optional[int] = Form(50),
    seed: Optional[int] = Form(None),
    response_format: Optional[str] = Form("b64_json")
):
    """
    OpenAI-compatible image editing endpoint
    
    Compatible with: POST /v1/images/edits
    """
    try:
        # Lazy load pipeline
        if not edit_loaded:
            print("First request - loading Edit pipeline...")
            load_edit_pipeline()
        
        if not edit_pipe:
            raise HTTPException(status_code=500, detail="Edit pipeline not loaded")
        
        if n > MAX_BATCH_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"n cannot exceed {MAX_BATCH_SIZE}"
            )
        
        # Clear memory before edit
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Read image from upload
        image_content = await image.read()
        pil_image = Image.open(io.BytesIO(image_content)).convert('RGB')
        
        # Create generator if seed is provided
        generator = None
        if seed is not None:
            generator = torch.Generator("cpu").manual_seed(seed)
        
        print(f"Editing image with prompt: {prompt[:50]}...")
        
        # Generate edited images
        with torch.inference_mode():
            output = edit_pipe(
                pil_image,
                prompt,
                negative_prompt=negative_prompt or "",
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                num_images_per_prompt=n,
                generator=generator
            )
        
        # Clear memory after edit
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Prepare response
        images_data = []
        for idx, image in enumerate(output.images):
            if response_format == "b64_json":
                images_data.append({
                    "b64_json": image_to_base64(image),
                    "index": idx
                })
            else:
                images_data.append({
                    "url": f"data:image/png;base64,{image_to_base64(image)}",
                    "index": idx
                })
        
        print("✓ Image editing completed")
        
        return {
            "created": int(torch.cuda.Event(enable_timing=True).record()),
            "data": images_data
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in image_edit: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "device": str(DEVICE),
        "cuda_available": torch.cuda.is_available(),
        "t2i_loaded": t2i_loaded,
        "edit_loaded": edit_loaded
    }


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "LongCat-Image API",
        "version": "1.0.0",
        "description": "OpenAI-compatible API for LongCat-Image",
        "endpoints": {
            "models": "/v1/models",
            "text_to_image": "/v1/images/generations",
            "image_edit": "/v1/images/edits",
            "health": "/v1/health",
            "docs": "/docs"
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv('API_PORT', 8000))
    host = os.getenv('API_HOST', '0.0.0.0')
    
    uvicorn.run(app, host=host, port=port, log_level="info")
