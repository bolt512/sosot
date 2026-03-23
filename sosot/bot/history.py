from collections import defaultdict, deque


class ChatHistory:
    """사용자별 in-memory 대화 히스토리."""

    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self._store: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=max_history)
        )

    def _key(self, channel_id: str, user_id: str) -> str:
        return f"{channel_id}:{user_id}"

    def add(self, channel_id: str, user_id: str, question: str, answer: str):
        self._store[self._key(channel_id, user_id)].append(
            {"question": question, "answer": answer}
        )

    def get(self, channel_id: str, user_id: str) -> list[dict]:
        return list(self._store[self._key(channel_id, user_id)])

    def format(self, channel_id: str, user_id: str) -> str:
        """프롬프트에 주입할 수 있는 문자열로 변환."""
        history = self.get(channel_id, user_id)
        if not history:
            return ""
        lines = []
        for turn in history:
            lines.append(f"사용자: {turn['question']}")
            lines.append(f"봇: {turn['answer']}")
        return "\n".join(lines)

    def clear(self, channel_id: str, user_id: str):
        self._store[self._key(channel_id, user_id)].clear()
