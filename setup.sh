#!/bin/bash

echo "🔧 Python 가상환경 생성 중..."
python3 -m venv venv
source venv/bin/activate

echo "📦 필수 패키지 설치 중..."
pip install --upgrade pip

# 기본 라이브러리
pip install \
    streamlit \
    python-dotenv \
    requests \
    beautifulsoup4

# LangChain 및 관련 의존성
pip install \
    langchain \
    langchain-openai \
    langchain-community \
    openai

# Tavily 검색 도구 (API 키 필요)
pip install tavily-python

echo "✅ 설치 완료!"

# .env 템플릿 생성
if [ ! -f .env ]; then
cat <<EOF > .env
OPENAI_API_KEY=your-openai-api-key
TAVILY_API_KEY=your-tavily-api-key
EOF
echo "📝 .env 파일을 생성했습니다. API 키를 입력해주세요!"
fi

# 프로젝트 시작 안내
echo "🚀 Streamlit 앱을 시작하려면 다음 명령어를 입력하세요:"
echo "source venv/bin/activate && streamlit run app.py"
