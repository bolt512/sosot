import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor

from mattermostdriver import Driver

from config.settings import Settings
from bot.chain import ask, build_rag_chain
from bot.history import ChatHistory
from bot.health import run_health_checks

logger = logging.getLogger(__name__)


class MattermostBot:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.chat_history = ChatHistory(max_history=settings.max_history)
        self.chain, self.retriever = build_rag_chain(settings, self.chat_history)
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.mm = Driver({
            "url": settings.mm_url,
            "port": settings.mm_port,
            "token": settings.mm_token,
            "scheme": settings.mm_scheme,
        })
        self.bot_id = None

    def start(self):
        """봇에 로그인하고 WebSocket 이벤트 수신을 시작한다."""
        self.mm.login()
        self.bot_id = self.mm.users.get_user_by_username(
            self.settings.bot_name
        )["id"]
        logger.info(f"봇 '{self.settings.bot_name}' 시작 (ID: {self.bot_id})")
        self.mm.init_websocket(self._handle_event)

    async def _handle_event(self, event):
        """WebSocket 이벤트를 처리한다."""
        if isinstance(event, str):
            event = json.loads(event)

        if event.get("event") != "posted":
            return

        try:
            post = json.loads(event["data"]["post"])
        except (KeyError, json.JSONDecodeError) as e:
            logger.warning(f"이벤트 파싱 오류: {e}")
            return

        message = post.get("message", "")
        user_id = post.get("user_id", "")
        channel_id = post.get("channel_id", "")
        post_id = post.get("id", "")

        # 봇 자신의 메시지 무시
        if user_id == self.bot_id:
            return

        # @멘션 확인
        mention = f"@{self.settings.bot_name}"
        if mention not in message:
            return

        query = message.replace(mention, "").strip()
        if not query:
            return

        logger.info(f"질문 수신 [채널:{channel_id}]: {query}")

        # 헬스체크 명령어
        if query == "!health":
            self._handle_health_command(channel_id, post_id)
            return

        # LangChain invoke를 스레드풀에서 실행 (블로킹 방지)
        loop = asyncio.get_event_loop()
        try:
            answer = await loop.run_in_executor(
                self.executor,
                ask,
                self.chain,
                self.chat_history,
                channel_id,
                query,
            )
        except Exception as e:
            logger.error(f"답변 생성 실패: {e}")
            answer = "죄송합니다. 답변 생성 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."

        self._send_reply(channel_id, post_id, answer)

    def _send_reply(self, channel_id: str, root_id: str, message: str):
        """Mattermost 채널에 답변을 전송한다."""
        try:
            self.mm.posts.create_post({
                "channel_id": channel_id,
                "message": message,
                "root_id": root_id,
            })
        except Exception as e:
            logger.error(f"메시지 전송 실패: {e}")

    def _handle_health_command(self, channel_id: str, root_id: str):
        """헬스체크 결과를 채팅으로 전송한다."""
        ok = run_health_checks(self.settings)
        status = "모든 시스템 정상" if ok else "일부 시스템에 문제가 있습니다. 로그를 확인하세요."
        self._send_reply(channel_id, root_id, f"**헬스체크 결과:** {status}")
