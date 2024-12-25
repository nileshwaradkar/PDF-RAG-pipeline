from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from prompts import RAG_PDF_PROMPT_TEMPLATE
from data.loader import load_pdf
from data.chunks import chunks
from embeddings import create_embeddings, create_vector_store


DOC_URL = "https://www.worldbank.org/content/dam/Worldbank/document/Climate/background-note_carbon-tax.pdf"


question = input("> Enter your query: ")

docs = load_pdf(DOC_URL)

all_splits = chunks(docs)

embeddings = create_embeddings("phi3:latest")

vector_store = create_vector_store(embeddings, all_splits)

model = ChatOllama(model="phi3:latest")

prompt = ChatPromptTemplate.from_template(RAG_PDF_PROMPT_TEMPLATE)


retrieved_docs = vector_store.similarity_search(question)

context = "\n\n".join([doc.page_content for doc in retrieved_docs])

prompt_query = prompt.format(question=question, context=context)

answer = model.invoke(prompt_query)

print(answer.content)