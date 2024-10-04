import os
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from PIL import Image
from io import BytesIO
from app.services.predictorService import process_and_predict
from datetime import datetime
import httpx
from typing import List

predict_router = APIRouter()

# Define directory for saving images
SAVE_DIR = "static/predictions/"
API_URL = os.getenv("API_URL", "https://api.smartwoundcare.site/api/wounds/type")

# Ensure the directory exists
os.makedirs(SAVE_DIR, exist_ok=True)


@predict_router.post("/predict/")
async def predict(file: UploadFile = File(None)):
    if file is None:
        return JSONResponse(
            content={"detail": "Image to predict not found"}, status_code=400
        )

    try:
        # Open the uploaded image
        image = Image.open(BytesIO(await file.read())).convert("RGB")

        # Process the image and get the results (multiple predictions)
        predictions, img_byte_arr = process_and_predict(image)

        # Generate a unique filename based on current timestamp
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"{timestamp}_prediction.jpg"
        file_path = os.path.join(SAVE_DIR, filename)

        # Save the image to the server
        with open(file_path, "wb") as f:
            f.write(img_byte_arr)

        # Extract wound types from predictions that are not "ไม่พบแผล"
        wound_types = [
            prediction["wound_type"]
            for prediction in predictions
            if prediction["wound_type"] != "ไม่พบแผล"
        ]

        additional_data_map = {}

        # If wound_types are present, fetch additional data from the external API
        if wound_types:
            payload = {"wound_types": wound_types}

            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(API_URL, json=payload)
                    response.raise_for_status()  # Raise exception for non-200 status codes

                    additional_data = response.json()

                    # Adjust to handle the new structure of wound_covers (which is an array of URLs)
                    for data in additional_data:
                        wound_type = data.get("wound_name")
                        if wound_type:
                            additional_data_map[wound_type] = {
                                "id": data["id"],
                                "wound_covers": data.get(
                                    "wound_covers", []
                                ),  # Handle array of wound_cover URLs
                            }
                        else:
                            print(
                                "Warning: Missing wound_name in the additional data response"
                            )

            except httpx.HTTPStatusError as e:
                # Handle response status errors (e.g., 4xx, 5xx)
                print(f"API response error: {e}")
            except httpx.RequestError as e:
                # Handle general request errors (e.g., connection problems)
                print(f"API request failed: {e}")
            except ValueError as e:
                # Handle data processing errors
                print(f"Data error: {e}")

        # Match additional data with each wound in predictions
        final_predictions = []
        for prediction in predictions:
            wound_type = prediction["wound_type"]

            # Include wound_covers from additional_data if available
            prediction_with_data = {
                "wound_type": wound_type,
                "confidence": prediction["confidence"],
                "additional_data": additional_data_map.get(
                    wound_type,
                    {
                        "id": None,
                        "wound_covers": [],  # Default to empty array if no data
                    },
                ),
            }

            final_predictions.append(prediction_with_data)

        # Prepare data for response
        response_data = {
            "predictions": final_predictions,
            "image_url": f"/predict/image/{filename}",
        }

        return JSONResponse(content=response_data)

    except Exception as e:
        # Return a 500 error with the error message
        error_message = f"Error processing the image: {str(e)}"
        return JSONResponse(content={"detail": error_message}, status_code=500)


@predict_router.get("/predict/image/{filename}")
async def get_predicted_image(filename: str):
    file_path = os.path.join(SAVE_DIR, filename)

    # Check if the file exists
    if not os.path.exists(file_path):
        return JSONResponse(content={"detail": "Image not found"}, status_code=404)

    # Return the saved image
    return FileResponse(file_path, media_type="image/png")
