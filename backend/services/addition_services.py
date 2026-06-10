class AdditionService:
    @staticmethod
    def add(a: int, b: int) -> dict:
        return {
            "result": a + b
        }