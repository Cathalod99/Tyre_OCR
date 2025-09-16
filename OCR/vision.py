from google.cloud import vision
import os

def detect_text(path):
    """Detects text in the file using Google Cloud Vision API."""
    # Check if credentials are set
    if not os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
        raise Exception("GOOGLE_APPLICATION_CREDENTIALS environment variable not set")
    
    try:
        client = vision.ImageAnnotatorClient()
        with open(path, 'rb') as image_file:
            content = image_file.read()
        image = vision.Image(content=content)
        response = client.text_detection(image=image)
        texts = response.text_annotations
        
        if response.error.message:
            raise Exception(f'Google Cloud Vision API error: {response.error.message}')
        
        if len(texts) == 0:
            return None
        
        return texts[0].description
    except Exception as e:
        print(f"⚠️ OCR Error: {e}")
        return None
