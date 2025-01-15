import streamlit as st
from transformers import pipeline
from PIL import Image
import io
import pytesseract

def main():
    st.title("Document QA System")
    
    # Configure pytesseract path for Windows (if needed)
    try:
        pytesseract.get_tesseract_version()
    except:
        # For Windows, you might need to set the path
        pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
    
    # Load the model
    @st.cache_resource
    def load_model():
        return pipeline(
            "document-question-answering",
            model="impira/layoutlm-document-qa"
        )
    
    try:
        nlp = load_model()
        st.success("Model loaded successfully!")
        
        # File uploader
        uploaded_file = st.file_uploader("Choose a document image", type=["png", "jpg", "jpeg"])
        
        if uploaded_file is not None:
            # Convert the uploaded file to PIL Image
            image_bytes = uploaded_file.read()
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert image to RGB if it's not
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Display image
            st.image(image, caption="Uploaded Document", use_column_width=True)
            
            # Question input
            question = st.text_input("Ask a question about the document:")
            
            if question and st.button("Get Answer"):
                with st.spinner("Processing..."):
                    try:
                        answer = nlp(image=image, question=question)
                        st.write("Answer:", answer)
                    except Exception as e:
                        st.error(f"Error processing question: {str(e)}")
                        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")

if __name__ == "__main__":
    main()