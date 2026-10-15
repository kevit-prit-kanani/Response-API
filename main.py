from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from response_api.app import app_router

main = FastAPI()

main.include_router(app_router)


@main.get("/")
def redirect_to_docs():
    return RedirectResponse(url="/docs")
