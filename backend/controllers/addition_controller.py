from fastapi import HTTPException
from services.addition_services import AdditionService

def add_numbers(a: int, b: int):
    res = AdditionService.add(a, b)
    if not res:
        raise HTTPException(status_code=400, detail="Addition failed")
    return res