"""
Examples for LongCat-Image OpenAI-Compatible API

Run the API server first:
    python api_server.py

Then run this script:
    python examples_api_usage.py
"""

import requests
import base64
import json
from PIL import Image
from io import BytesIO
import time

# ==================== Configuration ====================
API_BASE_URL = "http://localhost:8000"
HEADERS = {"Content-Type": "application/json"}


# ==================== Utility Functions ====================
def decode_image(b64_str: str, filename: str = "output.png"):
    """Decode base64 image and save to file"""
    img_data = base64.b64decode(b64_str)
    img = Image.open(BytesIO(img_data))
    img.save(filename)
    print(f"✅ Image saved: {filename}")
    return img


def print_response(response: dict, title: str = "Response"):
    """Pretty print API response"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    if 'data' in response:
        print(f"Created: {response.get('created', 'N/A')}")
        print(f"Images: {len(response['data'])}")
        for idx, img_data in enumerate(response['data']):
            print(f"  [{idx}] Index: {img_data.get('index', 'N/A')}")
            if 'b64_json' in img_data:
                print(f"       Format: Base64 (PNG, ~{len(img_data['b64_json'])} chars)")
            elif 'url' in img_data:
                print(f"       URL: {img_data['url'][:50]}...")
    else:
        print(json.dumps(response, indent=2))
    print(f"{'='*60}\n")


# ==================== Example 1: Health Check ====================
def example_health_check():
    """Check API health status"""
    print("\n🔍 Example 1: Health Check")
    print("-" * 60)
    
    try:
        response = requests.get(f"{API_BASE_URL}/v1/health", timeout=10)
        response.raise_for_status()
        data = response.json()
        
        print(f"Status: {data['status']}")
        print(f"Device: {data['device']}")
        print(f"CUDA Available: {data['cuda_available']}")
        print(f"T2I Loaded: {data['t2i_loaded']}")
        print(f"Edit Loaded: {data['edit_loaded']}")
        
        if not data['t2i_loaded'] or not data['edit_loaded']:
            print("\n⚠️  Warning: Not all pipelines are loaded!")
            return False
        
        print("\n✅ API is ready!")
        return True
    except requests.exceptions.ConnectionError:
        print("❌ Connection error! Make sure the API server is running.")
        print("   Run: python api_server.py")
        return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False


# ==================== Example 2: List Models ====================
def example_list_models():
    """List available models"""
    print("\n🤖 Example 2: List Available Models")
    print("-" * 60)
    
    try:
        response = requests.get(f"{API_BASE_URL}/v1/models", timeout=10)
        response.raise_for_status()
        data = response.json()
        
        print(f"Total models: {len(data['data'])}\n")
        for model in data['data']:
            print(f"  • {model['id']}")
            print(f"    Owner: {model['owned_by']}")
            print()
        
        return True
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False


# ==================== Example 3: Text-to-Image ====================
def example_text_to_image():
    """Generate image from text prompt"""
    print("\n🎨 Example 3: Text-to-Image Generation")
    print("-" * 60)
    
    prompt = "一只可爱的黑色猫咪，坐在粉红色的靠垫上，窗边的阳光。摄影风格，高质量，细节丰富。"
    
    print(f"Prompt: {prompt}\n")
    print("⏳ Generating image (this may take 20-30 seconds on GPU)...")
    
    try:
        start_time = time.time()
        
        response = requests.post(
            f"{API_BASE_URL}/v1/images/generations",
            json={
                "prompt": prompt,
                "negative_prompt": "ugly, distorted, blurry, bad quality",
                "n": 1,
                "size": "1344x768",
                "guidance_scale": 4.5,
                "num_inference_steps": 50,
                "seed": 42,
                "response_format": "b64_json"
            },
            headers=HEADERS,
            timeout=300  # 5 minute timeout
        )
        response.raise_for_status()
        
        elapsed = time.time() - start_time
        data = response.json()
        
        print_response(data, f"Text-to-Image Response ({elapsed:.1f}s)")
        
        # Save the image
        img_b64 = data['data'][0]['b64_json']
        decode_image(img_b64, "example_t2i_output.png")
        
        return True
    except requests.exceptions.Timeout:
        print("❌ Request timed out! Generation took too long.")
        return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False


# ==================== Example 4: Text-to-Image with Different Seeds ====================
def example_text_to_image_different_seeds():
    """Generate multiple images with different seeds"""
    print("\n🎨 Example 4: Text-to-Image with Different Seeds")
    print("-" * 60)
    
    prompt = "一个美丽的东方女人，穿着传统服装，古典场景"
    
    for seed in [42, 123, 456]:
        print(f"\nGenerating with seed {seed}...")
        
        try:
            response = requests.post(
                f"{API_BASE_URL}/v1/images/generations",
                json={
                    "prompt": prompt,
                    "n": 1,
                    "guidance_scale": 4.5,
                    "num_inference_steps": 30,
                    "seed": seed,
                    "response_format": "b64_json"
                },
                headers=HEADERS,
                timeout=300
            )
            response.raise_for_status()
            data = response.json()
            
            img_b64 = data['data'][0]['b64_json']
            decode_image(img_b64, f"example_t2i_seed_{seed}.png")
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            return False
    
    return True


# ==================== Example 5: Image Editing ====================
def example_image_editing():
    """Edit an image based on text prompt"""
    print("\n✏️  Example 5: Image Editing")
    print("-" * 60)
    
    # First, generate an image to edit
    print("Step 1: Generate base image...")
    try:
        response = requests.post(
            f"{API_BASE_URL}/v1/images/generations",
            json={
                "prompt": "一只可爱的橙色猫咪",
                "n": 1,
                "guidance_scale": 4.5,
                "num_inference_steps": 30,
                "seed": 999,
                "response_format": "b64_json"
            },
            headers=HEADERS,
            timeout=300
        )
        response.raise_for_status()
        base_img_b64 = response.json()['data'][0]['b64_json']
        base_img_data = base64.b64decode(base_img_b64)
        
        # Save base image
        with open("example_edit_base.png", "wb") as f:
            f.write(base_img_data)
        print("✅ Base image saved: example_edit_base.png")
        
    except Exception as e:
        print(f"❌ Error generating base image: {str(e)}")
        return False
    
    # Now edit the image
    print("\nStep 2: Edit the image...")
    edit_prompt = "把猫的颜色改成蓝色"
    
    try:
        with open("example_edit_base.png", "rb") as f:
            files = {'image': f}
            data = {
                'prompt': edit_prompt,
                'negative_prompt': 'ugly, distorted',
                'guidance_scale': 4.5,
                'num_inference_steps': 50,
                'seed': 42,
                'response_format': 'b64_json'
            }
            
            print(f"Edit prompt: {edit_prompt}\n")
            print("⏳ Editing image (this may take 20-30 seconds on GPU)...")
            
            response = requests.post(
                f"{API_BASE_URL}/v1/images/edits",
                files=files,
                data=data,
                timeout=300
            )
            response.raise_for_status()
        
        result = response.json()
        print_response(result, "Image Edit Response")
        
        # Save the edited image
        img_b64 = result['data'][0]['b64_json']
        decode_image(img_b64, "example_edit_output.png")
        
        return True
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False


# ==================== Example 6: Error Handling ====================
def example_error_handling():
    """Demonstrate error handling"""
    print("\n⚠️  Example 6: Error Handling")
    print("-" * 60)
    
    # Try with invalid size
    print("Testing invalid parameters...")
    try:
        response = requests.post(
            f"{API_BASE_URL}/v1/images/generations",
            json={
                "prompt": "test",
                "size": "invalid"  # Invalid size format
            },
            headers=HEADERS,
            timeout=10
        )
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        print(f"✅ Correctly caught HTTP error: {e.response.status_code}")
    except Exception as e:
        print(f"Error: {str(e)}")
    
    # Try with too many images
    print("\nTesting too many images...")
    try:
        response = requests.post(
            f"{API_BASE_URL}/v1/images/generations",
            json={
                "prompt": "test",
                "n": 100  # Too many
            },
            headers=HEADERS,
            timeout=10
        )
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        print(f"✅ Correctly caught error: {e.response.json()['detail']}")
    except Exception as e:
        print(f"Error: {str(e)}")
    
    return True


# ==================== Main ====================
def main():
    print("\n" + "="*60)
    print("  LongCat-Image OpenAI-Compatible API - Examples")
    print("="*60)
    
    # Step 1: Check health
    if not example_health_check():
        print("\n⚠️  API is not ready. Please start the server:")
        print("    python api_server.py")
        return
    
    # Step 2: List models
    example_list_models()
    
    # Step 3: Generate image from text
    example_text_to_image()
    
    # Step 4: Generate with different seeds (optional - slower)
    # example_text_to_image_different_seeds()
    
    # Step 5: Edit image
    example_image_editing()
    
    # Step 6: Error handling
    example_error_handling()
    
    print("\n" + "="*60)
    print("  ✅ All examples completed!")
    print("="*60)
    print("\n📖 For more information, see API.md")
    print("📚 Interactive API docs: http://localhost:8000/docs\n")


if __name__ == "__main__":
    main()
