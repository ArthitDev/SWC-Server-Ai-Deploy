from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config.cors import origins
from app.routes.predictRoutes import predict_router
from app.routes.mainRoutes import main_router
import os

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(main_router)
app.include_router(predict_router)


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
