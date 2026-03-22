from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

# Define the model
llm = HuggingFaceEndpoint(
    repo_id = 'meta-llama/Llama-3.1-8B-Instruct',
    task = 'text-generation'
)

model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser()

template = PromptTemplate(
    template = 'Give me the name, age and city of a fictional person \n {formate_instruction}',
    input_variables=[],
    partial_variables={'formate_instruction': parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({})

print(result)