# app/main.py
from fastapi import FastAPI
from app.controllers.query_controller import router as query_router
from app.config import settings

app = FastAPI(title="RAG Service")

# Registrar rutas
app.include_router(query_router, prefix="/query")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.HOST, port=settings.PORT)
