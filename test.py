from langchain_community.llms import HuggingFaceEndpoint
from config import api_token
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_community.document_loaders import TextLoader, PyPDFLoader


# Example 4 - PDF File
loader = TextLoader("./data/sc.txt")
# mydata = loader.load()
# print("My Data: ", mydata)
# print("My Data [0]: ", mydata[0])
# print("My Data [0]:\n", mydata[0].page_content)
# print("My Metadata: ", mydata[0].metadata)

my_context = loader.load()[0].page_content
human_template = "{question}\n{company_legal_doc}"
chat_prompt = ChatPromptTemplate.from_messages([
    HumanMessagePromptTemplate.from_template(human_template)
])
formatted_chat_prompt = chat_prompt.format_messages(
    question = "",
    company_legal_doc=my_context
)

repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
llm = HuggingFaceEndpoint(
    repo_id=repo_id, huggingfacehub_api_token=api_token
)

# print("Formatted Chat Prompt: ", formatted_chat_prompt)
# Streaming
response = llm.stream(formatted_chat_prompt)
for res in response:
    print(res, end="", flush=True)
