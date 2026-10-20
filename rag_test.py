
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Путь к файлу с методикой
methodology_path = Path(__file__).parent / "methodology.txt"

if not methodology_path.exists():
    raise FileNotFoundError(
        f"Не найден файл методики: {methodology_path}"
    )

text = methodology_path.read_text(encoding="utf-8")

# Разбиваем методику на небольшие фрагменты
splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=100,
)

documents = splitter.split_documents([
    Document(
        page_content=text,
        metadata={"source": "methodology.txt"},
    )
])

# Создаём эмбеддинги локальной моделью Ollama
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Строим векторный индекс
vector_store = FAISS.from_documents(documents, embeddings)

# Ищем фрагменты, относящиеся к запросу
query = "Как рассчитывается процент отклонения факта от плана?"
results = vector_store.similarity_search(query, k=2)

print("\n=== РЕЗУЛЬТАТ ПОИСКА RAG ===")

for i, document in enumerate(results, start=1):
    print(f"\nФрагмент {i}:")
    print(document.page_content)
    print("Источник:", document.metadata["source"])
