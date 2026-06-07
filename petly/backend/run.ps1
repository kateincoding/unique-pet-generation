# Levanta el backend Petly (FastAPI).
# Uso: .\run.ps1   (desde petly/backend)
if (-not $env:PETLY_MODELS_DIR) {
    $env:PETLY_MODELS_DIR = "c:/Users/Paulina Peralta/Desktop/gen-pet"
}
Write-Host "PETLY_MODELS_DIR = $env:PETLY_MODELS_DIR"
uvicorn app.main:app --reload --port 8000
