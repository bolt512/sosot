import logging
import os
import shutil

from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

from config.settings import Settings

logger = logging.getLogger(__name__)


def load_documents(data_path: str):
    """data_path에서 PDF, TXT, MD 파일을 로드한다."""
    documents = []

    loaders = [
        ("**/*.pdf", PyPDFLoader),
        ("**/*.txt", TextLoader),
        ("**/*.md", TextLoader),
    ]

    for glob_pattern, loader_cls in loaders:
        loader = DirectoryLoader(
            data_path,
            glob=glob_pattern,
            loader_cls=loader_cls,
            loader_kwargs={"encoding": "utf-8"} if loader_cls == TextLoader else {},
        )
        try:
            docs = loader.load()
            logger.info(f"{glob_pattern}: {len(docs)}개 문서 로드")
            documents.extend(docs)
        except Exception as e:
            logger.warning(f"{glob_pattern} 로드 중 오류: {e}")

    logger.info(f"총 {len(documents)}개 문서 로드 완료")
    return documents


def create_vector_db(settings: Settings | None = None) -> Chroma:
    """문서를 로드하고 청킹 후 벡터 DB를 생성한다."""
    if settings is None:
        settings = Settings.from_env()

    if not os.path.exists(settings.data_path):
        os.makedirs(settings.data_path)

    if os.path.exists(settings.db_path):
        shutil.rmtree(settings.db_path)
        logger.info(f"기존 벡터 DB 삭제: {settings.db_path}")

    documents = load_documents(settings.data_path)

    if not documents:
        logger.warning("로드된 문서가 없습니다. data/ 디렉토리에 파일을 추가하세요.")
        return None

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    texts = text_splitter.split_documents(documents)
    logger.info(f"{len(texts)}개 청크로 분할 완료")

    embeddings = OllamaEmbeddings(
        model=settings.embedding_model,
        base_url=settings.ollama_base_url,
    )

    vector_db = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=settings.db_path,
    )
    logger.info(f"벡터 DB 생성 완료: {settings.db_path}")
    return vector_db


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    create_vector_db()
