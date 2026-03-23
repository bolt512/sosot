import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, Request, HTTPException, Response
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
        self.rag_chain, self.retriever, self.rephrase_chain = build_rag_chain(settings, self.chat_history)
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.mm = Driver({
            "url": settings.mm_url,
            "port": settings.mm_port,
            "token": settings.mm_token,
            "scheme": settings.mm_scheme,
        })
        self.bot_id = None

    def login(self):
        """Mattermost에 로그인하고 봇 ID를 조회한다."""
        self.mm.login()
        self.bot_id = self.mm.users.get_user_by_username(
            self.settings.bot_name
        )["id"]
        logger.info(f"봇 '{self.settings.bot_name}' 로그인 완료 (ID: {self.bot_id})")

    def create_app(self) -> FastAPI:
        """FastAPI 앱을 생성하고 webhook 엔드포인트를 등록한다."""
        app = FastAPI()

        @app.post("/webhook")
        async def handle_webhook(request: Request):
            # Content-Type에 따라 JSON 또는 Form 파싱
            content_type = request.headers.get("content-type", "")
            if "application/json" in content_type:
                data = await request.json()
            else:
                form = await request.form()
                data = dict(form)

            token = data.get("token", "")
            text = data.get("text", "")
            user_id = data.get("user_id", "")
            channel_id = data.get("channel_id", "")
            post_id = data.get("post_id", "")

            logger.debug(f"수신된 post_id: '{post_id}', content-type: '{content_type}'")

            # 1) 토큰 검증
            if token != self.settings.webhook_token:
                raise HTTPException(status_code=403, detail="Invalid token")

            # 2) 봇 자신 메시지 무시
            if user_id == self.bot_id:
                return Response(status_code=200)

            # 3) 쿼리 추출
            query = text.strip()
            if not query:
                return Response(status_code=200)

            logger.info(f"질문 수신 [채널:{channel_id}, 사용자:{user_id}]: {query}")

            # 4) LLM 처리 (비동기)
            loop = asyncio.get_event_loop()
            try:
                answer = await loop.run_in_executor(
                    self.executor,
                    ask,
                    self.rag_chain,
                    self.retriever,
                    self.rephrase_chain,
                    self.chat_history,
                    channel_id,
                    user_id,
                    query,
                )
            except Exception as e:
                logger.error(f"답변 생성 실패: {e}")
                answer = "죄송합니다. 답변 생성 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."

            # 5) Driver API로 답변 전송
            self._send_reply(channel_id, post_id, answer)
            return Response(status_code=200)

        @app.get("/health")
        def health():
            ok = run_health_checks(self.settings)
            return {"status": "ok" if ok else "degraded"}

        return app

    def _send_reply(self, channel_id: str, root_id: str, message: str):
        """Mattermost 채널에 답변을 전송한다."""
        try:
            self.mm.posts.create_post({
                "channel_id": channel_id,
                "message": message,
                "root_id": root_id,
            })
            logger.info(f"답변 전송 완료 [채널:{channel_id}]")
        except Exception as e:
            logger.error(f"메시지 전송 실패: {e}")
