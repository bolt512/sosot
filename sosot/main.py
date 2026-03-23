import logging
import sys

import uvicorn

from config.settings import Settings
from bot.health import run_health_checks
from bot.handler import MattermostBot


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    logger.info("설정 로드 중...")
    settings = Settings.from_env()

    if not settings.mm_token:
        logger.error("MM_TOKEN이 설정되지 않았습니다. .env 파일을 확인하세요.")
        sys.exit(1)

    logger.info("헬스체크 실행 중...")
    if not run_health_checks(settings):
        logger.warning("일부 헬스체크가 실패했습니다. 계속 진행합니다...")

    logger.info("봇 시작 중...")
    bot = MattermostBot(settings)
    bot.login()
    app = bot.create_app()
    uvicorn.run(app, host="0.0.0.0", port=settings.webhook_port)


if __name__ == "__main__":
    main()
