import os
import hashlib
import dotenv
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from openai import OpenAI

dotenv.load_dotenv()

client = OpenAI(
    api_key=os.getenv('GROQ_API_KEY'),
    base_url='https://api.groq.com/openai/v1'
)

model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L12-v2",
    encode_kwargs={"normalize_embeddings": True}
)

vector_store = Chroma(
    collection_name="medical_data",
    embedding_function=model,
    persist_directory='./chroma',
    collection_metadata={'hnsw:space': 'cosine'}
)

def llm(query, context):
    system_prompt = f"""You are a medical information assistant. Answer the user's question using ONLY the provided source documents below.

        Rules you must follow:
        1. Base your answer ONLY on the provided sources - do not use outside knowledge
        2. If the sources don't contain enough information to answer, say exactly: "I don't have enough information to answer this question."
        3. Cite sources using [Source N] notation whenever you use information from them
        4. If sources contradict each other, mention the contradiction
        5. Be concise - answer in 3-5 sentences unless the question requires more detail
        6. Never speculate or make assumptions beyond what the sources say

        Sources:
        {context}"""
    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': query},
    ]
    response = client.chat.completions.create(
        model='openai/gpt-oss-20b',
        temperature=0.5,
        messages=messages
    )
    return response.choices[0].message.content


def generate_id(docs):
    return [hashlib.md5(doc.page_content.encode()).hexdigest() for doc in docs]


def read_pdf_file(file_name):
    reader = PdfReader(file_name)
    docs = []
    for index, page in enumerate(reader.pages):
        doc = Document(
            page_content=page.extract_text(),
            metadata={'source': file_name, 'page': index+1}
        )
        docs.append(doc)
    return docs


def main():
    docs = read_pdf_file('your_file.pdf')
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(docs)
    print(len(chunks))
    print(chunks[0])

    ids = generate_id(chunks)
    print(ids[0])

    vector_store.add_documents(chunks, ids=ids)
    all_docs = vector_store.get()
    print(len(all_docs['documents']))
    print(all_docs['documents'][0])

    while True:
        user_input = input(
            'ask something (type "q" to terminate): ').strip().lower()
        if user_input == 'q':
            print("good luck")
            break
            
        retrieved_chunks = vector_store.similarity_search_with_score(
            user_input, k=3)

        parts = []
        for i, (doc, score) in enumerate(retrieved_chunks):
            # اصلاح f-string و متد .get()
            parts.append(
                f"source{i+1} | page{doc.metadata.get('page', '?')} | cosine similarity(score: {score:.3f})\n{doc.page_content}"
            )

        context = '\n\n___\n\n'.join(parts)
        

        print(llm(user_input, context))


if __name__ == '__main__':
    main()
