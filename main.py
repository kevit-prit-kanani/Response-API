from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from response_api.app import app_router

main = FastAPI()

main.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

main.include_router(app_router)


@main.get("/")
def redirect_to_docs():
    return RedirectResponse(url="/docs")
