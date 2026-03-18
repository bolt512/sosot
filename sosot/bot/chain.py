import logging

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from config.settings import Settings
from bot.history import ChatHistory

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
당신은 회사 IT 헬프데스크 챗봇입니다. 사내 IT 매뉴얼을 바탕으로 직원들의 질문에 친절하고 정확하게 답변합니다.

규칙:
1. 제공된 컨텍스트 문서에 있는 정보만 사용하여 답변하세요.
2. 문서에 관련 정보가 없으면 "해당 내용은 제가 가진 매뉴얼에 없습니다. IT 담당자에게 문의해 주세요. (내선 1234)" 라고 안내하세요.
3. 답변은 간결하고 단계별로 작성하세요.
4. 한국어로 답변하세요.

이전 대화:
{history}

참고 문서:
{context}

질문: {question}"""


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(settings: Settings, chat_history: ChatHistory):
    """RAG 체인을 생성하여 (chain, retriever) 튜플을 반환한다."""
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

    chain = (
        {
            "context": (lambda x: x["question"]) | retriever | format_docs,
            "question": lambda x: x["question"],
            "history": lambda x: x["history"],
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever


def ask(chain, chat_history: ChatHistory, channel_id: str, question: str) -> str:
    """질문에 대한 답변을 생성하고 히스토리에 저장한다."""
    try:
        history_text = chat_history.format(channel_id)
        answer = chain.invoke({
            "question": question,
            "history": history_text,
        })
        chat_history.add(channel_id, question, answer)
        return answer
    except Exception as e:
        logger.error(f"체인 실행 오류: {e}")
        return "죄송합니다. 답변 생성 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."
