import os
import json
from dotenv import load_dotenv

import streamlit as st

# 크롤링
import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document

# LangChain Core
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# LangChain-OpenAI
from langchain_openai import ChatOpenAI

# Tavily 검색 도구
from langchain_community.tools.tavily_search import TavilySearchResults

# --- 환경 변수 불러오기 ---
load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- Streamlit UI 설정 ---
st.set_page_config(page_title="💰 비트코인 가격 예측 챗봇")
st.title("💰 RAG 기반 비트코인 가격 예측 챗봇 \n ex) 2025년 4월 한달간의 데이터를 봤을 시 5월달 비트코인의 가격 전망은 어떻게 될까")

# --- 세션 스테이트에 대화 기록 초기화 ---
if "history" not in st.session_state:
    st.session_state.history = []

# --- Tavily 검색 도구 초기화 ---
search_tool = TavilySearchResults(api_key=TAVILY_API_KEY)

def search_docs(query: str, k: int = 30) -> list[Document]:
    # Tavily 검색 결과
    results = search_tool.invoke(query)
    tavily_docs = [
        Document(page_content=entry["content"], metadata={"source": entry["url"]})
        for entry in results[:k]
    ]

    # mempool.space 크롤링 결과
    title, content, url = crawl_mempool_space()
    mempool_doc = Document(page_content=content, metadata={"source": url, "title": title})

    # 두 결과 합쳐서 반환
    return tavily_docs + [mempool_doc]

# --- 블록체인 mempool ---
def crawl_mempool_space():
    try:
        url = "https://mempool.space/"
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")

        # 비트코인 가격 추출 (예시)
        price_element = soup.find("span", class_="text-xl font-bold")  # 클래스명 확인 필요
        price = price_element.text.strip() if price_element else "N/A"

        # 블록 정보 추출 (예시)
        block_height_element = soup.find("a", {"href": lambda href: href and href.startswith("/block/")})
        block_height = block_height_element.text.strip() if block_height_element else "N/A"

        # 구글에서 비트코인 가격 크롤링
        google_price = crawl_google_bitcoin_price()

        # 코인마켓캡에서 비트코인 거래량 크롤링
        volume = crawl_coinmarketcap_volume()

        # 코인마켓캡에서 비트코인 RSI 크롤링
        rsi = crawl_coinmarketcap_rsi()

        content = f"현재 비트코인 가격: {price}, 최신 블록 높이: {block_height}, 구글에서 크롤링한 가격: {google_price}, 코인마켓캡에서 크롤링한 24시간 거래량: {volume}, 코인마켓캡에서 크롤링한 RSI: {rsi}"
        return "Bitcoin Information", content, url
    except Exception as e:
        print(f"Error crawling mempool.space: {e}")
        return "Error", "Could not retrieve Bitcoin information.", ""

# --- 구글에서 비트코인 가격 크롤링 ---
def crawl_google_bitcoin_price():
    try:
        url = "https://www.google.com/search?q=bitcoin+price"
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")

        # 예측 가격 범위 정보가 표시되는 요소 찾기
        range_element = soup.find("div", class_="range")  # 실제 클래스명 확인 필요
        if range_element:
            min_price, max_price = range_element.text.split("-")  # 범위 문자열 파싱
            min_price = min_price.strip()
            max_price = max_price.strip()
        else:
            min_price = "N/A"
            max_price = "N/A"

        return price, min_price, max_price  # 세 가지 값 반환
    except Exception as e:
        print(f"Error crawling Google for Bitcoin price: {e}")
        return "N/A"

# --- 코인마켓 캡에서의 비트코인 거래량 크롤링
def crawl_coinmarketcap_volume():
    try:
        url = "https://coinmarketcap.com/currencies/bitcoin/"
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")

        # 거래량 정보가 표시되는 요소 찾기 (예시 - 최신 웹 페이지 구조 확인 필요)
        volume_element = soup.find("div", class_="statsValue___2iaoZ")  # 클래스명 확인 필요
        if volume_element:
            volume = volume_element.text.strip()
        else:
            volume = "N/A"  # 거래량 정보를 찾지 못한 경우

        return volume
    except Exception as e:
        print(f"Error crawling CoinMarketCap for volume: {e}")
        return "N/A"

# --- 코인마켓 캡에서 비트코인 rsi 크롤링
def crawl_coinmarketcap_rsi():
    try:
        url = "https://coinmarketcap.com/currencies/bitcoin/technical-indicators/"
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")

        # RSI 정보가 표시되는 요소 찾기 (예시 - 최신 웹 페이지 구조 확인 필요)
        rsi_element = soup.find("span", string="RSI")  # RSI 텍스트를 포함하는 요소 찾기
        if rsi_element:
            rsi_value = rsi_element.find_next("span").text.strip()  # RSI 값 추출
        else:
            rsi_value = "N/A"  # RSI 정보를 찾지 못한 경우

        return rsi_value
    except Exception as e:
        print(f"Error crawling CoinMarketCap for RSI: {e}")
        return "N/A"


# --- Prompt 템플릿 정의 ---
prompt = PromptTemplate.from_template(
    """이전 대화:
{chat_history}

질문에 답변할 때, 아래 문맥을 참고하고 출처(URL)를 함께 인용해주세요.
특히 비트코인 가격 관련 질문은 아래 "Bitcoin Information" 문맥을 **반드시** 참조하고, 
다음과 같은 요소들을 **모두 포함**하여 답변해주세요.
가격을 말해 줄 때는 항상 달러($) 기준으로 말해주고 가격 앞에 $ 사인을 항상 붙여줘서 사람들이 어떤 돈의 기준인지 알 수 있게 해줘

## 비트코인 가격 예측 정보:

* **현재 구글링한 미국 주식시장 기준 비트코인 가격:** 현재 구글 기준 비트코인의 가격입니다: ${google_price} 달러
* **코인마켓캡에서 크롤링한 24시간 거래량:** 코인마켓캡에서 크롤링한 24시간 거래량 정보입니다: ${coinmarketcap_stat}.
* **코인마켓캡에서 크롤링한 RSI:** 코인마켓캡에서 크롤링한 RSI 정보입니다: ${coinmarketcap_rsi}

## 비트코인 가격 예측에 고려해야 할 요소:

**다음 순서대로 답변을 작성해주세요.**

1. **선 요약:** 다음 달 비트코인의 가격은 오늘보다 오를 것/내릴 것 입니다. 그 이유는 아래와 같습니다.
2. **공급과 수요:** 비트코인의 발행량은 제한되어 있으며, 수요 변화에 따라 가격이 영향을 받습니다. 
    * 예시: 최근 비트코인에 대한 기관 투자자들의 수요가 증가하고 있어 가격이 상승할 것으로 예상됩니다.
3. **글로벌 경제 상황:** 세계 경제 상황, 인플레이션, 금리 등 거시경제 요인이 비트코인 가격에 영향을 줄 수 있습니다.
    * 예시: 현재 미국 연준의 금리 인상 정책으로 인해 투자 심리가 위축되어 비트코인 가격이 하락할 가능성이 있습니다.
4. **규제 변화:** 각국의 암호화폐 규제 정책 변화는 비트코인 가격에 큰 영향을 미칩니다.
    * 예시: 중국 정부의 암호화폐 거래 금지 정책으로 인해 비트코인 가격이 급락한 사례가 있습니다.
5. **시장 심리:** 투자자들의 심리와 기대감은 비트코인 가격 변동에 중요한 역할을 합니다.
    * 예시: 일론 머스크의 트윗 하나로 비트코인 가격이 크게 변동하는 경우가 있습니다.
6. **기술적 분석:** 차트 패턴, 거래량, 이동 평균선 등 기술적 지표를 활용하여 가격 변동을 예측할 수 있습니다.
    * 예시: 비트코인 가격이 200일 이동 평균선을 상향 돌파하면 강세장으로 전환될 가능성이 높습니다.
7. **전쟁 및 정치적 불안정:** 실제로 일어나는 전쟁, 무역 전쟁, 국가 간의 정치적 갈등 등으로 인한 화폐 가치의 변화 등이 비트코인 가격에 큰 영향을 미칩니다.
    * 예시: 러시아-우크라이나 전쟁으로 인해 안전 자산 선호 현상이 강해지면서 금과 비트코인 가격이 상승했습니다.
8. **결론:** 위와 같은 내용을 기반으로 다음 달 비트코인의 가격은 현재 기준에서 오른다면/내린다면 최저 $ {min_price} 달러에서 최대 $ {max_price} 달러로 오를 것/내릴 것으로 예상됩니다. 


## 추가 정보:

* 예측은 참고용이며, 투자 결정에 대한 책임은 사용자에게 있습니다.
* 암호화폐 시장은 변동성이 매우 크므로, 투자 시 신중하게 판단해야 합니다.


문맥:
{context}

질문:
{question}
"""
)

# --- OpenAI LLM 설정 ---
llm = ChatOpenAI(model="gpt-4.1-mini", api_key=OPENAI_API_KEY)

# --- RAG 체인 구성 ---
def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

web_rag_chain = (
    {
        "context": lambda x: format_docs(search_docs(x["question"])),
        "question": RunnablePassthrough(),
        "chat_history": RunnablePassthrough(),
        "google_price": lambda x: crawl_google_bitcoin_price()[0],  # 구글 비트코인 가격 추가,
        "coinmarketcap_stat": lambda x: crawl_coinmarketcap_volume(), # 코인마켓캡 비트코인 거래량 추가 
        "coinmarketcap_rsi": lambda x: crawl_coinmarketcap_rsi(), # 코인마켓캡 비트코인 rsi 추가 
        "min_price": lambda x: crawl_google_bitcoin_price()[1],  # 최저 가격
        "max_price": lambda x: crawl_google_bitcoin_price()[2]   # 최고 가격
    }
    | prompt
    | llm
    | StrOutputParser()
)

# --- 대화 기록 출력 영역 ---
st.markdown("## 대화 기록")
for entry in st.session_state.history:
    if entry["role"] == "user":
        st.markdown(f"**You:** {entry['message']}")
    else:
        st.markdown(f"**Bot:** {entry['message']}")
st.markdown("---")

# --- 사용자 질문 입력 및 처리 ---
question = st.text_input("💬 질문을 입력하세요")
if question:
    # 1) 세션에 사용자 메시지 저장
    st.session_state.history.append({"role": "user", "message": question})

    # 2) Tavily 문서 검색 + RAG 응답 생성
    docs = search_docs(question)
    answer = web_rag_chain.invoke({"question": question, "chat_history": st.session_state.history})

    # 3) 세션에 봇 응답 저장
    st.session_state.history.append({"role": "bot", "message": answer})

    # 4) 대화 기록 파일로 영구 저장
    with open("history.json", "w", encoding="utf-8") as f:
        json.dump(st.session_state.history, f, ensure_ascii=False, indent=2)

    # 5) 응답 및 출처 출력
    sources = "\n".join(f"- {doc.metadata['source']}" for doc in docs)
    st.markdown(answer + "\n\n---\n**🔗 출처**\n" + sources)
