# LongCat-Image OpenAI-Compatible API

Un server API che espone le funzionalità di text-to-image generation e image editing di LongCat-Image con un'interfaccia compatibile con le API OpenAI.

## Features

- ✅ **Text-to-Image Generation** - Genera immagini da testo
- ✅ **Image Editing** - Modifica immagini esistenti
- ✅ **OpenAI Compatible** - Interfaccia compatibile con le API OpenAI
- ✅ **Base64 Output** - Immagini restituite in formato base64
- ✅ **GPU Support** - Supporto CUDA per accelerazione GPU
- ✅ **Docker Ready** - Pronto per containerizzazione

## Installazione

### Installazione locale

```bash
# Clonare il repository
git clone https://github.com/meituan-longcat/LongCat-Image
cd LongCat-Image

# Creare ambiente conda
conda create -n longcat-image python=3.10
conda activate longcat-image

# Installare dipendenze
pip install -r requirements.txt
pip install fastapi uvicorn python-multipart
python setup.py develop

# Download modelli
huggingface-cli download meituan-longcat/LongCat-Image --local-dir ./weights/LongCat-Image
huggingface-cli download meituan-longcat/LongCat-Image-Edit --local-dir ./weights/LongCat-Image-Edit

# Lanciare il server
python api_server.py
```

### Docker

```bash
# Build l'immagine Docker
docker build -t longcat-image-api:latest .

# Lanciare il container (assicurarsi di avere i modelli scaricati)
docker run --gpus all \
  -p 8000:8000 \
  -v /path/to/weights:/app/weights \
  longcat-image-api:latest
```

#### Con Docker Compose

```yaml
version: '3.8'

services:
  longcat-api:
    build: .
    image: longcat-image-api:latest
    container_name: longcat-image-api
    ports:
      - "8000:8000"
    volumes:
      - ./weights:/app/weights
    environment:
      - API_HOST=0.0.0.0
      - API_PORT=8000
      - USE_CPU_OFFLOAD=true
      - MAX_BATCH_SIZE=1
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

Eseguire: `docker-compose up -d`

## Variabili di Ambiente

- `API_HOST` (default: `0.0.0.0`) - Host del server API
- `API_PORT` (default: `8000`) - Porta del server API
- `T2I_CHECKPOINT` (default: `./weights/LongCat-Image`) - Path al modello text-to-image
- `EDIT_CHECKPOINT` (default: `./weights/LongCat-Image-Edit`) - Path al modello di editing
- `USE_CPU_OFFLOAD` (default: `true`) - Abilitare CPU offload per risparmiare VRAM
- `MAX_BATCH_SIZE` (default: `1`) - Numero massimo di immagini per richiesta

## API Endpoints

### 1. List Models
```
GET /v1/models
```

Lista i modelli disponibili.

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "longcat-image-t2i",
      "object": "model",
      "owned_by": "meituan-longcat"
    },
    {
      "id": "longcat-image-edit",
      "object": "model",
      "owned_by": "meituan-longcat"
    }
  ]
}
```

### 2. Text-to-Image Generation
```
POST /v1/images/generations
```

Genera immagini da un prompt testuale.

**Request Body:**
```json
{
  "prompt": "Una montagna innevata al tramonto",
  "negative_prompt": "brutto, distorto",
  "n": 1,
  "size": "1344x768",
  "guidance_scale": 4.5,
  "num_inference_steps": 50,
  "seed": 42,
  "response_format": "b64_json"
}
```

**Parameters:**
- `prompt` (string, required) - Descrizione dell'immagine da generare
- `negative_prompt` (string, optional) - Descrizione di ciò che NON deve essere presente
- `n` (integer, optional, default: 1) - Numero di immagini da generare (max 1)
- `size` (string, optional, default: "1344x768") - Dimensioni dell'immagine (WIDTHxHEIGHT)
- `guidance_scale` (float, optional, default: 4.5) - Forza del guidance (0-20)
- `num_inference_steps` (integer, optional, default: 50) - Numero di step di inferenza
- `seed` (integer, optional) - Seed per reproducibilità
- `response_format` (string, optional, default: "b64_json") - Formato risposta ("b64_json" o "url")

**Response:**
```json
{
  "created": 1700000000,
  "data": [
    {
      "b64_json": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
      "index": 0
    }
  ]
}
```

### 3. Image Editing
```
POST /v1/images/edits
```

Modifica un'immagine esistente in base a un prompt.

**Request (multipart/form-data):**
- `image` (file, required) - Immagine da modificare (PNG o JPEG)
- `prompt` (string, required) - Descrizione della modifica
- `negative_prompt` (string, optional) - Descrizione di ciò che NON deve essere presente
- `n` (integer, optional, default: 1) - Numero di immagini da generare
- `guidance_scale` (float, optional, default: 4.5) - Forza del guidance
- `num_inference_steps` (integer, optional, default: 50) - Numero di step di inferenza
- `seed` (integer, optional) - Seed per reproducibilità
- `response_format` (string, optional, default: "b64_json") - Formato risposta

**Response:**
```json
{
  "created": 1700000000,
  "data": [
    {
      "b64_json": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
      "index": 0
    }
  ]
}
```

### 4. Health Check
```
GET /v1/health
```

Verifica lo stato dell'API.

**Response:**
```json
{
  "status": "ok",
  "device": "cuda",
  "cuda_available": true,
  "t2i_loaded": true,
  "edit_loaded": true
}
```

## Esempi di Utilizzo

### Python

#### Text-to-Image
```python
import requests
import base64
from PIL import Image
from io import BytesIO

# Configurazione
API_URL = "http://localhost:8000"
HEADERS = {"Content-Type": "application/json"}

# Generare immagine da testo
response = requests.post(
    f"{API_URL}/v1/images/generations",
    json={
        "prompt": "Una gatta nera che dorme su un cuscino rosa",
        "negative_prompt": "brutto, sfocato, distorto",
        "n": 1,
        "guidance_scale": 4.5,
        "num_inference_steps": 50,
        "seed": 123
    },
    headers=HEADERS
)

# Decodificare e salvare l'immagine
data = response.json()
img_b64 = data['data'][0]['b64_json']
img_data = base64.b64decode(img_b64)
img = Image.open(BytesIO(img_data))
img.save('generated_image.png')
```

#### Image Editing
```python
import requests
import base64
from PIL import Image
from io import BytesIO

API_URL = "http://localhost:8000"

# Caricare e modificare un'immagine
with open('original_image.png', 'rb') as f:
    files = {'image': f}
    data = {
        'prompt': 'Cambia il gatto in un cane',
        'guidance_scale': 4.5,
        'num_inference_steps': 50
    }
    
    response = requests.post(
        f"{API_URL}/v1/images/edits",
        files=files,
        data=data
    )

# Decodificare e salvare l'immagine modificata
result = response.json()
img_b64 = result['data'][0]['b64_json']
img_data = base64.b64decode(img_b64)
img = Image.open(BytesIO(img_data))
img.save('edited_image.png')
```

### cURL

#### Text-to-Image
```bash
curl -X POST http://localhost:8000/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Una montagna innevata",
    "n": 1,
    "size": "1344x768"
  }' | python -m json.tool > response.json
```

#### Image Editing
```bash
curl -X POST http://localhost:8000/v1/images/edits \
  -F "image=@original_image.png" \
  -F "prompt=Cambia il colore in blu" \
  -F "guidance_scale=4.5" | python -m json.tool > response.json
```

### JavaScript/Node.js

#### Text-to-Image
```javascript
const fetch = require('node-fetch');

const response = await fetch('http://localhost:8000/v1/images/generations', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    prompt: "Un tramonto rosso su una spiaggia",
    n: 1,
    guidance_scale: 4.5,
    num_inference_steps: 50
  })
});

const data = await response.json();
const base64Image = data.data[0].b64_json;

// Salvare l'immagine
const fs = require('fs');
fs.writeFileSync('image.png', Buffer.from(base64Image, 'base64'));
```

#### Image Editing
```javascript
const FormData = require('form-data');
const fs = require('fs');
const fetch = require('node-fetch');

const form = new FormData();
form.append('image', fs.createReadStream('original.png'));
form.append('prompt', 'Aggiungi un tramonto');
form.append('guidance_scale', 4.5);

const response = await fetch('http://localhost:8000/v1/images/edits', {
  method: 'POST',
  body: form
});

const data = await response.json();
const base64Image = data.data[0].b64_json;
fs.writeFileSync('edited.png', Buffer.from(base64Image, 'base64'));
```

## Accesso alla Documentazione Interattiva

Una volta lanciato il server, accedere a:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Requisiti di Sistema

### CPU
- Processore moderno con supporto AVX2

### GPU (Consigliato)
- NVIDIA GPU con compute capability >= 7.0 (Volta o successivo)
- Almeno 20 GB VRAM per inferen completezza

### RAM
- Almeno 32 GB di RAM disponibile (con CPU offload)

### Storage
- ~50 GB per i modelli scaricati

## Troubleshooting

### Errore: "CUDA out of memory"
- Assicurarsi che `USE_CPU_OFFLOAD=true` sia impostato
- Ridurre `MAX_BATCH_SIZE` a 1
- Usare GPU con più memoria

### Errore: "Modelli non trovati"
- Verificare che i percorsi in `T2I_CHECKPOINT` e `EDIT_CHECKPOINT` siano corretti
- Assicurarsi che i modelli siano stati scaricati con:
  ```bash
  huggingface-cli download meituan-longcat/LongCat-Image --local-dir ./weights/LongCat-Image
  huggingface-cli download meituan-longcat/LongCat-Image-Edit --local-dir ./weights/LongCat-Image-Edit
  ```

### Slow inference
- Se non si usa GPU, i tempi di inferenza saranno molto lunghi (30+ minuti per immagine)
- Usare una GPU NVIDIA con supporto CUDA

## Performance

Su una NVIDIA A100 (40GB VRAM):
- Text-to-Image (50 steps): ~15-20 secondi
- Image Editing (50 steps): ~15-20 secondi

Con CPU offload su GPU consumer:
- Text-to-Image (50 steps): ~20-30 secondi
- Image Editing (50 steps): ~20-30 secondi

## Note Importanti

### Rendering di Testo
Per ottenere il miglior rendering di testo nelle immagini, **racchiudere il testo tra virgolette**:

```python
# Corretto ✅
prompt = 'Una persona che indossa una maglietta con scritto \"Hello World\"'

# Non ideale ❌
prompt = 'Una persona che indossa una maglietta con scritto Hello World'
```

### Seed per Reproducibilità
Usare lo stesso `seed` con gli stessi parametri produrrà sempre la stessa immagine:

```python
# Entrambe le chiamate genereranno la stessa immagine
requests.post(f"{API_URL}/v1/images/generations", json={
    "prompt": "Un gatto",
    "seed": 42
})
```

## License

Questo progetto segue la stessa licenza del progetto LongCat-Image.

## Support

Per problemi o domande:
- GitHub Issues: https://github.com/meituan-longcat/LongCat-Image/issues
- Discussioni: https://github.com/meituan-longcat/LongCat-Image/discussions
