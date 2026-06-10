from fastapi import APIRouter
from controllers.addition_controller import add_numbers

router = APIRouter(prefix="/api", tags=["operations"])

@router.post("/add")
async def add_endpoint(a: int, b: int):
    return add_numbers(a, b)
