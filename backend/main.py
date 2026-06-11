from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.detection import router as detection_router

app = FastAPI(title="SkinLens API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(detection_router)

@app.get("/")
async def root():
    return {"message": "Welcome to SkinLens API"}

if __name__ == "__main__":
    from configs.ngrok import ngrok_config
    ngrok_config()
    
    from pyngrok import ngrok
    import uvicorn

    try:
        public_url = ngrok.connect(8000, "http")
        print(f"ngrok tunnel URL: {public_url}")
    except Exception as e:
        print(f"Note: ngrok error - {e}")
        print("Continuing without ngrok tunnel...")

    uvicorn.run(app, host="localhost", port=8000)
