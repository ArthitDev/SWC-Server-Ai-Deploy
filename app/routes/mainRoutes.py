from fastapi import APIRouter

main_router = APIRouter()


@main_router.get("/")
def getMainInfo():
    return {
        "message": "Welcome to the Wound Prediction API!",
        "endpoints": {
            "/predict/": "POST an image file to this endpoint to get the wound prediction."
        },
    }
