import logging

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config.settings import Settings
from bot.history import ChatHistory

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
당신은 회사 IT 헬프데스크 챗봇입니다. 사내 IT 매뉴얼을 바탕으로 직원들의 질문에 친절하고 정확하게 답변합니다.

규칙:
1. 제공된 컨텍스트 문서에 있는 정보만 사용하여 답변하세요.
2. 문서에 관련 정보가 없으면 다음과 같이 안내하세요:
   "해당 내용은 제가 가진 매뉴얼에 없습니다. IT 담당자에게 직접 문의해 주세요. (내선 1234 / it-help@company.com)"
3. 답변은 **마크다운 형식**으로 작성하세요:
   - 단계가 있는 경우 번호 목록 사용
   - 중요 정보는 **굵게** 표시
   - 경로나 명령어는 `코드 블록` 사용
4. 답변은 간결하게, 꼭 필요한 정보만 포함하세요.
5. 한국어로 답변하세요.
6. 이전 대화를 참고하여 맥락에 맞게 답변하세요.

이전 대화:
{history}

참고 문서:
{context}

질문: {question}"""

REPHRASE_PROMPT = """\
다음 대화 히스토리와 후속 질문이 주어졌을 때,
후속 질문을 대화 맥락 없이도 이해할 수 있는 완전한 질문으로 바꿔주세요.
원래 질문의 의미를 유지하고, 한국어로 작성하세요.

대화 히스토리:
{history}

후속 질문: {question}

재작성된 질문:"""


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(settings: Settings, chat_history: ChatHistory):
    """RAG 체인을 생성하여 (chain, retriever, llm) 튜플을 반환한다."""
    embeddings = OllamaEmbeddings(
        model=settings.embedding_model,
        base_url=settings.ollama_base_url,
    )
    vectorstore = Chroma(
        persist_directory=settings.db_path,
        embedding_function=embeddings,
    )
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": settings.retriever_k}
    )
    llm = ChatOllama(
        model=settings.llm_model,
        base_url=settings.ollama_base_url,
        temperature=0,
    )
    prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)
    rephrase_prompt = ChatPromptTemplate.from_template(REPHRASE_PROMPT)

    rephrase_chain = rephrase_prompt | llm | StrOutputParser()

    rag_chain = prompt | llm | StrOutputParser()

    return rag_chain, retriever, rephrase_chain


def ask(rag_chain, retriever, rephrase_chain, chat_history: ChatHistory, channel_id: str, user_id: str, question: str) -> str:
    """질문에 대한 답변을 생성하고 히스토리에 저장한다."""
    try:
        history_text = chat_history.format(channel_id, user_id)

        # 히스토리가 있으면 질문을 standalone으로 rewrite하여 검색에 사용
        if history_text:
            search_query = rephrase_chain.invoke({
                "history": history_text,
                "question": question,
            })
            search_query = search_query.strip()
            logger.debug(f"질문 재작성: '{question}' → '{search_query}'")
        else:
            search_query = question

        # retriever 직접 호출
        docs = retriever.invoke(search_query)
        context = format_docs(docs)

        # 답변 생성
        answer = rag_chain.invoke({
            "question": question,
            "history": history_text,
            "context": context,
        })
        chat_history.add(channel_id, user_id, question, answer)
        return answer
    except Exception as e:
        logger.error(f"체인 실행 오류: {e}")
        return "죄송합니다. 답변 생성 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."
