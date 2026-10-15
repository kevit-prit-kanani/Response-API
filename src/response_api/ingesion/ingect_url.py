from typing import Annotated

from fastapi import APIRouter, Body

ingect_url = APIRouter(
    prefix="ingesion",
    tags=['ingesion']
)


@ingect_url.post('/ingect')
def fetch_data_from_url(url: Annotated[str, Body()]):
    return url
