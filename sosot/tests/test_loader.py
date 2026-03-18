import os
import tempfile
from unittest.mock import patch, MagicMock

from ingest.loader import load_documents, create_vector_db
from config.settings import Settings


def test_load_txt_documents():
    """TXT 파일이 정상적으로 로드되는지 확인."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # TXT 파일 생성
        txt_path = os.path.join(tmpdir, "test.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("테스트 문서 내용입니다.")

        docs = load_documents(tmpdir)
        assert len(docs) == 1
        assert "테스트 문서 내용" in docs[0].page_content


def test_load_empty_directory():
    """빈 디렉토리에서 빈 리스트를 반환하는지 확인."""
    with tempfile.TemporaryDirectory() as tmpdir:
        docs = load_documents(tmpdir)
        assert docs == []


def test_load_multiple_files():
    """여러 TXT 파일이 모두 로드되는지 확인."""
    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(3):
            path = os.path.join(tmpdir, f"doc_{i}.txt")
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"문서 {i}번 내용")

        docs = load_documents(tmpdir)
        assert len(docs) == 3


@patch("ingest.loader.Chroma")
@patch("ingest.loader.OllamaEmbeddings")
def test_create_vector_db(mock_embeddings, mock_chroma):
    """벡터 DB 생성이 올바르게 호출되는지 확인."""
    mock_chroma.from_documents.return_value = MagicMock()

    with tempfile.TemporaryDirectory() as tmpdir:
        # 테스트용 파일 생성
        txt_path = os.path.join(tmpdir, "test.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("테스트 문서 내용입니다. " * 50)

        db_path = os.path.join(tmpdir, "db")
        settings = Settings(
            mm_url="localhost", mm_port=8065, mm_token="test",
            mm_scheme="http", bot_name="sosot",
            ollama_base_url="http://localhost:11434",
            llm_model="llama3", embedding_model="nomic-embed-text",
            db_path=db_path, data_path=tmpdir,
            chunk_size=500, chunk_overlap=50,
            retriever_k=3, max_history=10,
        )

        result = create_vector_db(settings)
        assert result is not None
        mock_chroma.from_documents.assert_called_once()
