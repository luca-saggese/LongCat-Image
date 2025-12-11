#!/bin/bash

# Simple script to call LongCat-Image API and generate images
# Server: 192.168.1.20:8000

API_URL="http://192.168.1.20:8000"
OUTPUT_DIR="./generated_images"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Default prompt
PROMPT="${1:-一只可爱的黑色猫咪，坐在粉红色的靠垫上，窗边的阳光。摄影风格，高质量，细节丰富。}"

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}LongCat-Image API Client${NC}"
echo -e "${BLUE}================================${NC}"
echo ""
echo -e "Server: ${GREEN}${API_URL}${NC}"
echo -e "Prompt: ${GREEN}${PROMPT}${NC}"
echo ""

# Check if server is reachable
echo -e "${BLUE}Checking server health...${NC}"
if ! curl -s "$API_URL/v1/health" > /dev/null; then
    echo -e "${RED}Error: Cannot reach API server at ${API_URL}${NC}"
    echo -e "Make sure the server is running at 192.168.1.20:8000"
    exit 1
fi
echo -e "${GREEN}✓ Server is online${NC}\n"

# Generate image
echo -e "${BLUE}Generating image...${NC}"
echo -e "${BLUE}This may take 20-30 seconds...${NC}\n"

RESPONSE=$(curl -s -X POST "$API_URL/v1/images/generations" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "'"$PROMPT"'",
    "negative_prompt": "ugly, distorted, blurry, bad quality",
    "n": 1,
    "size": "1344x768",
    "guidance_scale": 4.5,
    "num_inference_steps": 50,
    "response_format": "b64_json"
  }')

# Check if response is valid JSON
if ! echo "$RESPONSE" | grep -q '"b64_json"'; then
    echo -e "${RED}Error generating image:${NC}"
    echo "$RESPONSE" | python -m json.tool 2>/dev/null || echo "$RESPONSE"
    exit 1
fi

# Extract base64 image
BASE64_IMG=$(echo "$RESPONSE" | python -c "import sys, json; data = json.load(sys.stdin); print(data['data'][0]['b64_json'])" 2>/dev/null)

if [ -z "$BASE64_IMG" ]; then
    echo -e "${RED}Error: Could not extract image from response${NC}"
    exit 1
fi

# Save image
OUTPUT_FILE="$OUTPUT_DIR/longcat_${TIMESTAMP}.png"
echo "$BASE64_IMG" | base64 -d > "$OUTPUT_FILE"

if [ -f "$OUTPUT_FILE" ]; then
    FILE_SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)
    echo -e "${GREEN}✓ Image generated successfully!${NC}"
    echo -e "${GREEN}  File: ${OUTPUT_FILE}${NC}"
    echo -e "${GREEN}  Size: ${FILE_SIZE}${NC}"
    echo ""
    echo -e "${BLUE}Quick preview (macOS):${NC}"
    echo -e "  ${GREEN}open ${OUTPUT_FILE}${NC}"
else
    echo -e "${RED}Error: Failed to save image${NC}"
    exit 1
fi
