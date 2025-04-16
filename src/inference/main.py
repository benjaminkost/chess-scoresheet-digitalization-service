from fastapi import FastAPI
from controllers import imageController

# Create FastAPI server
app = FastAPI()

# Include routing to access domain-specific controllers
app.include_router(imageController.image_controller)

