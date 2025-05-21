import socket

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from .controllers import imageController

def get_ip():
    return socket.gethostbyname(socket.gethostname())

# Server IP
SERVER_IP = get_ip()

# Create FastAPI server
app = FastAPI()

# Include routing to access domain-specific controllers
app.include_router(imageController.image_controller)

# Manage access regarding: CORS (Access-controll-allow-origin), etc.
origins = [
    f"http://{SERVER_IP}",
    "http://localhost:5173",
    "http://localhost:9000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

