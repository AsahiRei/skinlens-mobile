from fastapi import HTTPException
from services.multiply_services import MultiplyService

def multiply_numbers(a: int, b: int):
    res = MultiplyService.multiply(a, b)
    if not res:
        raise HTTPException(status_code=400, detail="Multiplication failed")
    return res