from fastapi import APIRouter
from controllers.multiply_controller import multiply_numbers

router = APIRouter(prefix="/api", tags=["operations"])

@router.post("/multiply")
async def multiply_endpoint(a: int, b: int):
    return multiply_numbers(a, b)