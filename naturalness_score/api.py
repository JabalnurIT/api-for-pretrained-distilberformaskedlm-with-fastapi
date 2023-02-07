# from typing import Dict

from fastapi import Depends, FastAPI
from pydantic import BaseModel

from .classifier.model import Model, get_model

app = FastAPI()


class NaturalnessRequest(BaseModel):
    text: str

    class Config:
        orm_mode = True


class NaturalnessMaskResponse(BaseModel):
    text_mask_fill: str


class NaturalnessPerplexityResponse(BaseModel):
    perplexity: float


@app.post("/fill_mask", response_model=NaturalnessMaskResponse)
async def fill_mask(request: NaturalnessRequest, model: Model = Depends(get_model)):
    text_mask_fill = model.fill_mask(request.text)
    return NaturalnessMaskResponse(
        text_mask_fill=text_mask_fill
    )


@app.post("/perplexity", response_model=NaturalnessPerplexityResponse)
async def perplexity(request: NaturalnessRequest, model: Model = Depends(get_model)):
    perplexity = model.perplexity(request.text)
    return NaturalnessPerplexityResponse(
        perplexity=perplexity
    )
