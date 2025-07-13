from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from bert_model import analyze_message  # Asegúrate que este archivo exista y esté en el mismo directorio

# Inicialización de la app FastAPI
app = FastAPI(
    title="Abuse Analyzer API",
    description="API to analyze messages and extract typologies, red flags, and abuse techniques.",
    version="1.0.0"
)

# Permitir CORS para frontend local (ajustar si es necesario)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Puedes restringirlo a ['http://localhost:3000'] si deseas
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelo de entrada para la API
class InputText(BaseModel):
    responses: str

# Ruta principal de prueba
@app.get("/")
def root():
    return {"message": "Abuse Analyzer API is running."}

# Ruta de análisis
@app.post("/analyze")
def analyze(input_data: InputText):
    try:
        result = analyze_message(input_data.responses)
        return result
    except Exception as e:
        print("❌ Internal server error:", e)
        raise HTTPException(status_code=500, detail="Internal Server Error")
