from unittest.mock import patch, MagicMock

from bot.history import ChatHistory
from bot.chain import ask, format_docs


def test_chat_history_add_and_get():
    """히스토리 추가/조회가 올바른지 확인."""
    history = ChatHistory(max_history=5)
    history.add("ch1", "Wi-Fi 어떻게 연결하나요?", "CORP-WIFI를 선택하세요.")

    items = history.get("ch1")
    assert len(items) == 1
    assert items[0]["question"] == "Wi-Fi 어떻게 연결하나요?"
    assert items[0]["answer"] == "CORP-WIFI를 선택하세요."


def test_chat_history_max_limit():
    """max_history 초과 시 오래된 항목이 제거되는지 확인."""
    history = ChatHistory(max_history=3)
    for i in range(5):
        history.add("ch1", f"질문{i}", f"답변{i}")

    items = history.get("ch1")
    assert len(items) == 3
    assert items[0]["question"] == "질문2"


def test_chat_history_format():
    """히스토리 포맷 문자열 확인."""
    history = ChatHistory(max_history=5)
    history.add("ch1", "질문1", "답변1")

    formatted = history.format("ch1")
    assert "사용자: 질문1" in formatted
    assert "봇: 답변1" in formatted


def test_chat_history_format_empty():
    """빈 히스토리의 포맷은 빈 문자열."""
    history = ChatHistory(max_history=5)
    assert history.format("ch1") == ""


def test_format_docs():
    """문서 포맷 함수 확인."""
    doc1 = MagicMock()
    doc1.page_content = "문서1 내용"
    doc2 = MagicMock()
    doc2.page_content = "문서2 내용"

    result = format_docs([doc1, doc2])
    assert "문서1 내용" in result
    assert "문서2 내용" in result
    assert "\n\n" in result


def test_ask_error_handling():
    """체인 실행 오류 시 안내 메시지를 반환하는지 확인."""
    mock_chain = MagicMock()
    mock_chain.invoke.side_effect = Exception("LLM 오류")

    history = ChatHistory(max_history=5)
    answer = ask(mock_chain, history, "ch1", "테스트 질문")

    assert "오류가 발생했습니다" in answer


def test_ask_saves_history():
    """정상 답변 시 히스토리에 저장되는지 확인."""
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = "답변입니다"

    history = ChatHistory(max_history=5)
    answer = ask(mock_chain, history, "ch1", "테스트 질문")

    assert answer == "답변입니다"
    items = history.get("ch1")
    assert len(items) == 1
    assert items[0]["question"] == "테스트 질문"
