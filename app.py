from transformers import pipeline
from PIL import Image
from pdf2image import convert_from_path  # For PDFs
import os

try:
    # Load the model
    nlp = pipeline(
        "document-question-answering",
        model="impira/layoutlm-document-qa"
    )
    print("Model loaded successfully!")

    # Example questions you can ask
    questions = [
        "What is the invoice number?",
        "What is the total amount?",
        "What is the date?",
        "Who is the sender?"
    ]

    # Replace with your document path (PDF or image)
    document_path = "your_document.pdf"  # or .png, .jpg

    # Handle PDF or image
    if document_path.endswith('.pdf'):
        # Convert first page of PDF to image
        images = convert_from_path(document_path, first_page=1, last_page=1)
        image = images[0]
    else:
        # Load image directly
        image = Image.open(document_path)

    # Ask each question
    for question in questions:
        answer = nlp(image=image, question=question)
        print(f"\nQuestion: {question}")
        print(f"Answer: {answer}")

except Exception as e:
    print(f"Error: {e}")
