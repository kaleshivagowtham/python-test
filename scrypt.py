from paddleocr import PaddleOCR, draw_ocr
from PIL import Image, ImageFilter
import cv2
import spacy

# Load PaddleOCR model
ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)

# Load spaCy model
nlp = spacy.load("en_core_web_lg")

def extract_text_and_entities(img_path):
    # Read image using OpenCV
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

    # Perform OCR using PaddleOCR
    result = ocr.ocr(img_path)
    result = result[0]  # Assuming only one page is processed

    # Extract text, bounding boxes, and confidence scores
    boxes = [line[0] for line in result]
    txts = [line[1][0] for line in result]
    scores = [line[1][1] for line in result]

    # Extract named entities using spaCy
    detected_text = []
    for t in txts:
        doc = nlp(t)
        named_entities = [(ent.text, ent.label_) for ent in doc.ents]
        detected_text.append(named_entities)

    # Determine which text boxes contain sensitive information
    indexes = []
    for i in range(len(detected_text)):
        if len(detected_text[i]) != 0:
            text = detected_text[i][0]
            if text[1] in ['DATE', 'GPE', 'PERSON', 'NORP', 'FAC', 'LOC', 'PRODUCT']:
                indexes.append(1)
            else:
                indexes.append(0)
        else:
            indexes.append(0)

    return img, boxes, indexes

def blur_sensitive_regions(img, boxes, indexes):
    # Convert OpenCV image to PIL image
    image = Image.fromarray(img)

    # Blur sensitive regions
    for i in range(len(boxes)):
        if indexes[i] == 1:
            box_coordinates = [boxes[i][0], boxes[i][2]]
            top_left = (int(box_coordinates[0][0]), int(box_coordinates[0][1]))
            bottom_right = (int(box_coordinates[1][0]), int(box_coordinates[1][1]))
            roi = image.crop((top_left[0], top_left[1], bottom_right[0], bottom_right[1]))
            blurred_roi = roi.filter(ImageFilter.GaussianBlur(radius=20))
            image.paste(blurred_roi, (top_left[0], top_left[1]))

    return image

if __name__ == "__main__":
    # Example usage
    img_path = './testImg.png'
    img, boxes, indexes = extract_text_and_entities(img_path)
    blurred_img = blur_sensitive_regions(img, boxes, indexes)
    blurred_img.show()
