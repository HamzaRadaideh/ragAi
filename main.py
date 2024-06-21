import os
import ast
import logging
from dotenv import load_dotenv
from pydantic import BaseModel
from llama_index.llms.ollama import Ollama
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.core.query_pipeline import QueryPipeline
from llama_parse import LlamaParse
from prompts import context, code_parser_template
from code_reader import code_reader

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize LLMs
llm = Ollama(model="llama3", request_timeout=30.0)
code_llm = Ollama(model="codellama")

# Initialize parser and document reader
parser = LlamaParse(result_type="markdown")
file_extractor = {".pdf": parser}
documents = SimpleDirectoryReader("./data", file_extractor=file_extractor).load_data()

# Initialize embedding model and vector index
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
vector_index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
query_engine = vector_index.as_query_engine(llm=llm)

# Set up tools for the agent
tools = [
    QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="api_documentation",
            description="This tool provides documentation about code for an API. Use it for reading docs for the API.",
        ),
    ),
    code_reader,
]

# Initialize agent
agent = ReActAgent.from_tools(tools, llm=code_llm, verbose=True, context=context)

# Define the output format
class CodeOutput(BaseModel):
    code: str
    description: str
    filename: str

# Set up output parser and pipeline
parser = PydanticOutputParser(CodeOutput)
json_prompt_str = parser.format(code_parser_template)
json_prompt_tmpl = PromptTemplate(json_prompt_str)
output_pipeline = QueryPipeline(chain=[json_prompt_tmpl, llm])

# Main interaction loop
while (prompt := input("Enter a prompt (q to quit): ")) != "q":
    retries = 0

    while retries < 3:
        try:
            result = agent.query(prompt)
            next_result = output_pipeline.run(response=result)
            cleaned_json = ast.literal_eval(str(next_result).replace("assistant:", ""))
            break
        except Exception as e:
            retries += 1
            logger.error(f"Error occurred, retry #{retries}: {e}", exc_info=True)

    if retries >= 3:
        logger.error("Unable to process request after 3 retries, try again...")
        continue

    logger.info("Code generated")
    print(cleaned_json["code"])
    print("\n\nDescription:", cleaned_json["description"])

    filename = cleaned_json["filename"]

    try:
        os.makedirs("output", exist_ok=True)
        with open(os.path.join("output", filename), "w") as f:
            f.write(cleaned_json["code"])
        logger.info(f"Saved file {filename}")
    except Exception as e:
        logger.error("Error saving file:", exc_info=True)