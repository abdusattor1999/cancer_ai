from fastapi import FastAPI, File, UploadFile
from fastapi.encoders import jsonable_encoder
from brest_cancer import router as breast_router

import os, io

app = FastAPI()


@app.get("/")
async def echo_handler():
    return {"message": "Ishla yaxshimi !!!"}



app.include_router(breast_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


