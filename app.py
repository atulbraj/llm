import streamlit as st
from transformers import pipeline
import pandas as pd

# Initialize the question-answering pipeline
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

def parse_document(document_text):
    """
    Parse the uploaded document into a structured Q&A format.
    Assume each main question and its follow-ups are separated by specific markers.
    """
    lines = document_text.splitlines()
    qa_data = []
    current_question = None
    current_answers = []
    
    for line in lines:
        if line.startswith("Main Question:"):
            # Save the previous question and answers
            if current_question:
                qa_data.append({"question": current_question, "answers": current_answers})
            current_question = line.replace("Main Question:", "").strip()
            current_answers = []
        elif line.startswith("Follow-up Question:"):
            follow_up = line.replace("Follow-up Question:", "").strip()
            current_answers.append(follow_up)
        elif line.startswith("Answer:"):
            answer = line.replace("Answer:", "").strip()
            current_answers.append(answer)
    
    # Append the last question if exists
    if current_question:
        qa_data.append({"question": current_question, "answers": current_answers})
    
    return qa_data

def find_best_response(question, qa_data):
    """
    Use NLP techniques to find the best response from the parsed document.
    """
    best_response = None
    best_score = -1

    for entry in qa_data:
        for answer in entry['answers']:
            # Evaluate each answer against the question
            result = qa_pipeline({'question': question, 'context': answer})
            if result['score'] > best_score:
                best_score = result['score']
                best_response = result['answer']
    
    return best_response

# Streamlit app UI
st.title("Personality-Based Q&A Analyzer")
st.write("Upload a Q&A document and ask questions based on its content.")

# File uploader
uploaded_file = st.file_uploader("Upload your Q&A document (.txt)", type=["txt"])

if uploaded_file:
    # Read and display the document content
    document_text = uploaded_file.read().decode("utf-8")
    st.write("### Uploaded Document")
    st.text_area("Document Content", document_text, height=300)
    
    # Parse the document
    qa_data = parse_document(document_text)
    
    # Display parsed Q&A
    st.write("### Parsed Q&A")
    df = pd.DataFrame(qa_data)
    st.dataframe(df)
    
    # Question input
    user_question = st.text_input("Ask a question:")
    
    if user_question:
        # Find the best response
        response = find_best_response(user_question, qa_data)
        
        if response:
            st.write("### Response")
            st.write(response)
        else:
            st.write("No relevant answer found in the document.")
