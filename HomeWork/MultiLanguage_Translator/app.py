from flask import Flask, render_template, request, jsonify
from langchain import PromptTemplate, LLMChain
from langdetect import detect
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Define the prompt template for translation
template = """
You are a professional translator. Translate the following text from {source_language} to {target_language}:
"{text}"

{explanation_request}
"""
prompt = PromptTemplate(
    input_variables=["text", "source_language", "target_language", "explanation_request"],
    template=template,
)

# Set up a Gemini chat model
llm = ChatGoogleGenerativeAI(
    model="gemini-pro", temperature=0.2, api_key=os.getenv("GOOGLE_API_KEY")
)

# Create the LLM chain
translation_chain = LLMChain(prompt=prompt, llm=llm)

# Detect the source language dynamically
def detect_language(text):
    try:
        return detect(text)
    except Exception as e:
        print(f"Error detecting language: {e}")
        return "unknown"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/translate", methods=["POST"])
def translate():
    # Get data from the frontend
    text = request.form.get("text")
    target_language = request.form.get("target_language")
    want_explanation = request.form.get("want_explanation") == "true"

    # Add explanation request if needed
    explanation_request = (
        "Explain the meaning of the translated text in context."
        if want_explanation
        else ""
    )

    # Detect the source language
    source_language = detect_language(text)
    if source_language == "unknown":
        return jsonify({"error": "Could not detect source language. Please try again."})

    # Run the translation chain
    try:
        response = translation_chain.run(
            {
                "text": text,
                "source_language": source_language,
                "target_language": target_language,
                "explanation_request": explanation_request,
            }
        )
        return jsonify(
            {"source_language": source_language, "translation": response}
        )
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
