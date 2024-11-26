from langchain_core.prompts import PromptTemplate
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define a prompt template
template = "You are a helpful assistant. Answer the following question: {question}"
prompt = PromptTemplate(
    input_variables=["question"],
    template=template
)

# Initialize the OpenAI LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-pro", temperature=0.2, api_key=os.getenv("GOOGLE_API_KEY")
)

# Combine the prompt template and the LLM into a chain
chain = prompt | llm

# Define an input question and run the chain
question = "What is LangChain in short and conscise way?"
response = chain.invoke(question)

# Print the response
print("Response from LLM:", response.content)
