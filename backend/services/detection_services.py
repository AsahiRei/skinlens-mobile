from utils.detection import detect_skin_condition, batch_detect
from PIL import Image
from io import BytesIO

class DetectionService:
    @staticmethod
    def detect_from_file(file_path: str, threshold: float = 0.5) -> dict:
        """
        Detect skin condition from a file path.
        
        Args:
            file_path: Path to image file
            threshold: Confidence threshold (0-1)
            
        Returns:
            Detection result dictionary
        """
        try:
            result = detect_skin_condition(file_path, threshold=threshold)
            return {
                "success": True,
                "data": result
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    @staticmethod
    def detect_from_bytes(image_bytes: bytes, threshold: float = 0.5) -> dict:
        """
        Detect skin condition from image bytes.
        
        Args:
            image_bytes: Image file as bytes
            threshold: Confidence threshold (0-1)
            
        Returns:
            Detection result dictionary
        """
        try:
            image = Image.open(BytesIO(image_bytes))
            result = detect_skin_condition(image, threshold=threshold)
            return {
                "success": True,
                "data": result
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    @staticmethod
    def detect_multiple(file_paths: list, threshold: float = 0.5) -> dict:
        """
        Detect skin conditions from multiple images.
        
        Args:
            file_paths: List of image file paths
            threshold: Confidence threshold (0-1)
            
        Returns:
            List of detection results
        """
        try:
            results = batch_detect(file_paths, threshold=threshold)
            return {
                "success": True,
                "data": results,
                "count": len(results)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
