import streamlit as st
from transformers import pipeline
from PIL import Image

def main():
    st.title("Document QA System")
    
    # Load the model
    @st.cache_resource  # This will cache the model
    def load_model():
        return pipeline(
            "document-question-answering",
            model="impira/layoutlm-document-qa"
        )
    
    nlp = load_model()
    st.success("Model loaded successfully!")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a document image", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Document", use_column_width=True)
        
        # Question input
        question = st.text_input("Ask a question about the document:")
        
        if question and st.button("Get Answer"):
            answer = nlp(image=image, question=question)
            st.write("Answer:", answer)

if __name__ == "__main__":
    main()