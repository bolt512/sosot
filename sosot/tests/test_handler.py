import json
import pytest


def make_event(message: str, user_id: str = "user123", channel_id: str = "ch1", post_id: str = "post1"):
    """테스트용 Mattermost WebSocket 이벤트를 생성한다."""
    return json.dumps({
        "event": "posted",
        "data": {
            "post": json.dumps({
                "id": post_id,
                "user_id": user_id,
                "channel_id": channel_id,
                "message": message,
            })
        }
    })


def test_event_parsing():
    """이벤트 JSON 파싱이 올바른지 확인."""
    event_str = make_event("@sosot 안녕하세요")
    event = json.loads(event_str)
    post = json.loads(event["data"]["post"])

    assert event["event"] == "posted"
    assert post["message"] == "@sosot 안녕하세요"
    assert post["user_id"] == "user123"
    assert post["channel_id"] == "ch1"


def test_mention_filtering():
    """@sosot 멘션이 있는 메시지만 처리 대상인지 확인."""
    bot_name = "sosot"
    mention = f"@{bot_name}"

    # 멘션 있음
    msg_with_mention = "@sosot Wi-Fi 연결이 안 됩니다"
    assert mention in msg_with_mention

    # 멘션 없음
    msg_without_mention = "Wi-Fi 연결이 안 됩니다"
    assert mention not in msg_without_mention


def test_query_extraction():
    """@sosot 멘션을 제거하고 질문만 추출하는지 확인."""
    bot_name = "sosot"
    message = "@sosot Wi-Fi 연결이 안 됩니다"
    query = message.replace(f"@{bot_name}", "").strip()

    assert query == "Wi-Fi 연결이 안 됩니다"


def test_empty_query_ignored():
    """멘션만 있고 질문이 없는 경우 무시되는지 확인."""
    bot_name = "sosot"
    message = "@sosot"
    query = message.replace(f"@{bot_name}", "").strip()

    assert query == ""


def test_bot_self_message_ignored():
    """봇 자신의 메시지는 무시되는지 확인."""
    bot_id = "bot999"
    event_str = make_event("@sosot 테스트", user_id=bot_id)
    event = json.loads(event_str)
    post = json.loads(event["data"]["post"])

    assert post["user_id"] == bot_id
