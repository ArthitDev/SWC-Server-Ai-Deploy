import os
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO
from app.services.predictorService import process_and_predict
from datetime import datetime
import httpx
from typing import List


load_dotenv()

predict_router = APIRouter()

# API URLs
API_URL = os.getenv("API_URL", "http://localhost:3000/api/wounds/type")
API_URL_UPLOAD = os.getenv("API_URL_UPLOAD", "http://localhost:3000/api/user/uploads")

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

        # Prepare the image to be sent to the external API (ExpressJS)
        files = {'file': ('image.jpg', img_byte_arr, 'image/jpeg')}
        async with httpx.AsyncClient() as client:
            upload_response = await client.post(API_URL_UPLOAD, files=files)
            upload_response.raise_for_status()
            upload_data = upload_response.json()

        # Get the uploaded image URL from the external API response
        uploaded_image_url = upload_data.get('image_url')

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
                    response.raise_for_status()

                    additional_data = response.json()

                    # Adjust to handle the new structure of wound_covers (which is an array of URLs)
                    for data in additional_data:
                        wound_type = data.get("wound_name")
                        if wound_type:
                            additional_data_map[wound_type] = {
                                "id": data["id"],
                                "wound_covers": data.get("wound_covers", []),
                            }
                        else:
                            print("Warning: Missing wound_name in the additional data response")

            except httpx.HTTPStatusError as e:
                print(f"API response error: {e}")
            except httpx.RequestError as e:
                print(f"API request failed: {e}")
            except ValueError as e:
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
                    {"id": None, "wound_covers": []},  # Default to empty array if no data
                ),
            }

            final_predictions.append(prediction_with_data)

        # Prepare data for response
        response_data = {
            "predictions": final_predictions,
            "image_url": uploaded_image_url,  # Return the uploaded image URL from ExpressJS
        }

        return JSONResponse(content=response_data)

    except Exception as e:
        # Return a 500 error with the error message
        error_message = f"Error processing the image: {str(e)}"
        return JSONResponse(content={"detail": error_message}, status_code=500)
