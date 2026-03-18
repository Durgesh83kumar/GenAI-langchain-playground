from dotenv import load_dotenv
import os
load_dotenv()

print("TOKEN:", os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"))

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

llm = HuggingFaceEndpoint(
    repo_id = "meta-llama/Llama-3.1-8B-Instruct",
    task = "text_generation"
)
model = ChatHuggingFace(llm=llm)

result = model.invoke("What is the capital of India")

print(result.content)