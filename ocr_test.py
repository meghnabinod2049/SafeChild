from PIL import Image

import pytesseract

from app import analyze_text


pytesseract.pytesseract.tesseract_cmd = r"C:\Users\meghn\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"


image = Image.open("chat.png")


text = pytesseract.image_to_string(image)


print("\nEXTRACTED TEXT:\n")

print(text)


result = analyze_text(text)


print("\nSAFECHILD ANALYSIS:\n")

print(result)