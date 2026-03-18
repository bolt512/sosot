import logging

import requests

from config.settings import Settings

logger = logging.getLogger(__name__)


def check_ollama(settings: Settings) -> bool:
    try:
        resp = requests.get(f"{settings.ollama_base_url}/api/tags", timeout=5)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        for needed in [settings.llm_model, settings.embedding_model]:
            if not any(needed in m for m in models):
                logger.warning(f"Ollama 모델 '{needed}'이(가) 설치되어 있지 않습니다.")
        logger.info("Ollama 연결 정상")
        return True
    except Exception as e:
        logger.error(f"Ollama 연결 실패: {e}")
        return False


def check_chromadb(settings: Settings) -> bool:
    import os
    if os.path.exists(settings.db_path):
        logger.info("ChromaDB 디렉토리 확인 완료")
        return True
    else:
        logger.warning(f"ChromaDB 디렉토리가 없습니다: {settings.db_path}")
        logger.warning("먼저 'python -m ingest.loader'를 실행하여 벡터 DB를 생성하세요.")
        return False


def check_mattermost(settings: Settings) -> bool:
    try:
        url = f"{settings.mm_scheme}://{settings.mm_url}:{settings.mm_port}/api/v4/system/ping"
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        logger.info("Mattermost 연결 정상")
        return True
    except Exception as e:
        logger.error(f"Mattermost 연결 실패: {e}")
        return False


def run_health_checks(settings: Settings) -> bool:
    """모든 헬스체크를 실행하고 전체 통과 여부를 반환한다."""
    results = [
        ("Ollama", check_ollama(settings)),
        ("ChromaDB", check_chromadb(settings)),
        ("Mattermost", check_mattermost(settings)),
    ]
    all_ok = all(ok for _, ok in results)
    if all_ok:
        logger.info("모든 헬스체크 통과")
    else:
        failed = [name for name, ok in results if not ok]
        logger.warning(f"헬스체크 실패: {', '.join(failed)}")
    return all_ok
