from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.encoders import jsonable_encoder
from pathlib import Path
from main import tf, model, save_and_display_gradcam, make_gradcam_heatmap, find_last_conv_layer, CURRENT_DIR
import shutil
import os, io
import cv2

app = FastAPI()
UPLOAD_DIR = Path("upload_images")
PROCESSED_IMAGES = Path("processed_images")

CANCER_THRESHOLD_MEASURES = {
    "no": "No cancer detected, everything is ok !",
    "low": "There are some unclear signs, and it’s uncertain whether these indicate cancer. Further diagnostic tests or a consultation with a specialist is necessary",
    "medium": "The images show suspicious areas that could indicate the presence of cancer. Please consult with a healthcare provider as soon as possible for a more thorough examination",
    "strong": "The images indicate a high likelihood of cancer. Immediate medical attention is required. Please consult an oncologist or your healthcare provider for the next steps.",
}

def get_description_message(threshold):
    key = "no"
    if 0 < threshold <= 0.2:
        key = "low"
    elif 0.2 < threshold <= 0.4:
        key = "medium"
    else:
        key = "strong"
    return CANCER_THRESHOLD_MEASURES[key]


def save_and_display_gradcam(img_path):
    img = cv2.imread(img_path)
    # Encode the processed image in memory
    success, encoded_image = cv2.imencode('.jpg', img)
    if not success:
        raise ValueError("Could not encode the image")

    # Return the image as a BytesIO object
    return io.BytesIO(encoded_image.tobytes())

def breast_cancer_test(image_path):
    processed_img_io = save_and_display_gradcam(image_path)
    decription = get_description_message(.4)
    return processed_img_io, decription


@app.post("/cancer-test/")
async def process_photo(file: UploadFile = File(...)):
    # Save the uploaded image to a temporary location
    img_path = f"upload_images/{file.filename}"
    with open(img_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Process the image using test_is_cancer
    processed_img_io, description = breast_cancer_test(img_path)

    # Clean up the uploaded file after processing
    try:
        os.remove(img_path)
    except Exception as e:
        print(f"Error deleting file: {e}")
    # Return both image and description
    return StreamingResponse(
        content=processed_img_io,
        media_type="image/jpeg",
        headers={"Description": description}
    )

@app.get("/")
async def echo_handler():
    return {"message": "Ishla yaxshimi !!!"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
