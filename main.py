import os
from dotenv import load_dotenv
import requests
from requests.auth import HTTPBasicAuth
import math
from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
LARAVEL_BASE_URL = os.getenv("LARAVEL_BASE_URL", "http://127.0.0.1:8000/api/v1/rag")
RAG_USERNAME = os.getenv("RAG_USERNAME", "rag_service")
RAG_PASSWORD = os.getenv("RAG_PASSWORD", "secret123")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY belum diatur di file .env")

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
BASIC_AUTH = HTTPBasicAuth(RAG_USERNAME, RAG_PASSWORD)
DOCS_DIR = "./Raw RAG Document"


def load_local_sop_documents(folder_path: str) -> list[Document]:
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
        return []

    pdf_docs = DirectoryLoader(folder_path, glob="**/*.pdf", loader_cls=PyPDFLoader).load()
    txt_docs = DirectoryLoader(folder_path, glob="**/*.txt", loader_cls=TextLoader).load()
    return pdf_docs + txt_docs


def fetch_ticket_knowledge_documents(since: str | None = None, category_id: int | None = None) -> list[Document]:
    docs: list[Document] = []
    page = 1
    limit = 100

    while True:
        params = {"limit": limit, "page": page}
        if since:
            params["since"] = since
        if category_id:
            params["category_id"] = category_id

        resp = requests.get(
            f"{LARAVEL_BASE_URL}/knowledge",
            headers={"Accept": "application/json"},
            params=params,
            auth=BASIC_AUTH,
            timeout=30,
        )
        resp.raise_for_status()
        payload = resp.json()

        data_obj = payload.get("data", {})
        items = data_obj.get("list", []) if isinstance(data_obj, dict) else []

        if not items:
            break

        for row in items:
            if not isinstance(row, dict):
                continue

            content = (
                f"Nomor Tiket: {row.get('ticket_number')}\n"
                f"Kategori: {row.get('category') or '-'} | Prioritas: {row.get('priority') or '-'}\n"
                f"Judul: {row.get('title')}\n"
                f"Keluhan/Deskripsi: {row.get('description')}\n"
                f"Solusi/Resolusi: {row.get('resolution')}"
            )
            metadata = {
                "source": f"ticket:{row.get('ticket_number')}",
                "ticket_id": row.get("ticket_id"),
                "category": row.get("category"),
                "department": row.get("department"),
                "updated_at": row.get("updated_at"),
            }
            docs.append(Document(page_content=content, metadata=metadata))

        meta = payload.get("meta", {}) if isinstance(payload.get("meta"), dict) else {}
        current_page = int(meta.get("page", page))
        total = int(meta.get("total", len(items)))
        per_page = int(meta.get("limit", limit)) or limit
        last_page = max(1, math.ceil(total / per_page))

        if current_page >= last_page:
            break
        page += 1

    return docs
def send_rag_webhook(ticket_id: int, action: str, content: str) -> dict:
    resp = requests.post(
        f"{LARAVEL_BASE_URL}/webhook",
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
        json={
            "ticket_id": ticket_id,
            "action": action,
            "content": content,
        },
        auth=BASIC_AUTH,
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


def main():
    print("1. Memuat dokumen SOP lokal...")
    sop_docs = load_local_sop_documents(DOCS_DIR)

    print("2. Mengambil knowledge tickets dari Laravel...")
    ticket_docs = fetch_ticket_knowledge_documents()

    all_docs = sop_docs + ticket_docs
    if not all_docs:
        raise ValueError("Tidak ada dokumen SOP maupun tiket resolved/closed yang tersedia.")

    print(f"Total dokumen: {len(all_docs)} (SOP: {len(sop_docs)}, Tiket: {len(ticket_docs)})")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(all_docs)

    print("3. Membangun Chroma Vector DB...")
    embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_metadata={"hnsw:space": "cosine"},
    )

    base_llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.1)
    llm = base_llm.with_retry(
    stop_after_attempt=5,
    wait_exponential_jitter=True,
)
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Anda adalah asisten chatbot IT Helpdesk.
Jawab pertanyaan pengguna HANYA berdasarkan konteks SOP dan riwayat tiket resolved di bawah ini.
Jika tidak ada solusi yang jelas di konteks, balas dengan "ESKALASI_DIBUTUHKAN".

Konteks:
{context}"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])
    chain = prompt | llm | StrOutputParser()
    chat_history = []

    print("\nChatbot RAG siap! Ketik 'exit' untuk keluar.\n")
    while True:
        question = input("Anda: ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break

        results = vector_db.similarity_search_with_score(question, k=3)
        relevant = [(doc, 1.0 - float(dist)) for doc, dist in results if (1.0 - float(dist)) >= 0.45]

        if not relevant:
            answer = "ESKALASI_DIBUTUHKAN"
        else:
            context_str = "\n\n---\n\n".join(
                f"[{doc.metadata.get('source')} | sim={sim:.2%}]\n{doc.page_content}"
                for doc, sim in relevant
            )
            answer = chain.invoke({
                "context": context_str,
                "chat_history": chat_history[-6:],
                "question": question,
            }).strip()

        chat_history.append(HumanMessage(content=question))
        chat_history.append(AIMessage(content=answer))
        print(f"Bot: {answer}\n")


if __name__ == "__main__":
    main()