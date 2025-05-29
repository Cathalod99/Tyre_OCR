from google.cloud import vision

import os

def detect_text(path):
    """Detects text in the file."""
    client = vision.ImageAnnotatorClient()
    with open(path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations
    
    if response.error.message:
        raise Exception(f'{response.error.message}')
    
    if len(texts) == 0:
        return None
    
    return texts[0].description
