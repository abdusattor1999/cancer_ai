from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.encoders import jsonable_encoder
from pathlib import Path
from main import tf, model, save_and_display_gradcam, make_gradcam_heatmap, find_last_conv_layer, CURRENT_DIR
import shutil
import os, io

webapp = FastAPI()
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

def test_is_cancer(image_path):
    # Load and preprocess the image
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(1024, 512))
    img_array = tf.keras.preprocessing.image.img_to_array(img)

    # Normalize the image if required by the model
    img_array = img_array / 255.0  # If the model was trained on normalized data
    img_array = tf.expand_dims(img_array, axis=0)  # Add batch dimension

    last_conv_layer_name = find_last_conv_layer(model)

    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)

    processed_img_io = save_and_display_gradcam(image_path, heatmap)
    predictions = model.predict(img_array)


    return processed_img_io, get_description_message(predictions[0])


@webapp.post("/cancer-test/")
async def process_photo(file: UploadFile = File(...)):
    # Save the uploaded image to a temporary location
    file_path = f"upload_images/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Process the image using test_is_cancer
    processed_img_io, description = test_is_cancer(file_path)

    # Clean up the uploaded file after processing
    try:
        os.remove(file_path)
    except Exception as e:
        print(f"Error deleting file: {e}")
    # Return both image and description
    return StreamingResponse(
        content=processed_img_io,
        media_type="image/jpeg",
        headers={"Description": description}
    )

@webapp.get("/aloo")
async def echo_handler():
    return {"message": "Ishla yaxshimi !!!"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(webapp, host="0.0.0.0", port=8000)


