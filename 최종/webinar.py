import os
from typing import List, Dict, Optional
import yt_dlp
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from openai import OpenAI
import tempfile
import streamlit as st
from streamlit_chat import message
import time
from datetime import datetime

# streamlit run webinar.py 실행

class WebinarSummarizer:
    def __init__(self, 
                 openai_api_key: Optional[str] = None):
        """
        Args:
            openai_api_key: OpenAI API 키
        """
        self.realtime_summary = []
        self.last_summary_time = 0
        self.summary_interval = 300  # 5분마다 요약
        
        # OpenAI API 키 설정
        if openai_api_key is None:
            openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OpenAI API 키가 필요합니다.")
        self.openai_api_key = openai_api_key
        self.client = OpenAI(api_key=openai_api_key)

        # 모델 초기화
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=self.openai_api_key
        )
        self.llm = ChatOpenAI(
            temperature=0,
            openai_api_key=self.openai_api_key
        )
            
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # qa_chain 초기화
        self.qa_chain = None
        self.vectorstore = None

    def extract_audio(self, video_path: str) -> str:
        """비디오 파일에서 오디오를 추출하고 STT로 변환합니다."""
        try:
            # 비디오 파일을 직접 Whisper API로 전송
            with open(video_path, "rb") as video:
                transcript = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=video,
                    language="ko"
                )
            return transcript.text
        except Exception as e:
            raise Exception(f"비디오 처리 중 오류 발생: {str(e)}")
        
    def transcribe_audio(self, audio_path: str) -> str:
        """오디오 파일을 텍스트로 변환합니다 (OpenAI Whisper API 사용)."""
        try:
            with open(audio_path, "rb") as audio_file:
                transcript = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language="ko"
                )
            os.remove(audio_path)  # 임시 오디오 파일 삭제
            return transcript.text
        except Exception as e:
            if os.path.exists(audio_path):
                os.remove(audio_path)
            raise Exception(f"오디오 인식 중 오류 발생: {str(e)}")
        
    def get_transcript(self, source: str) -> str:
        """YouTube URL 또는 비디오 파일에서 자막을 추출합니다."""
        if source.startswith(("http://", "https://")):
            return self.download_transcript(source)
        else:
            # 비디오 파일 처리
            return self.extract_audio(source)
        
    def download_transcript(self, url: str) -> str:
        """YouTube 영상에서 오디오를 추출하고 STT로 변환합니다."""
        ydl_opts = {
            'format': 'bestaudio/best',
            'quiet': False,
            'no_warnings': False,
            'outtmpl': '%(id)s.%(ext)s',
            'writesubtitles': False,
            'writeautomaticsub': False,
            'subtitleslangs': [],
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'nocheckcertificate': True,
            'ignoreerrors': True,
            'no_color': True,
            'extractor_args': {
                'youtube': {
                    'player_client': ['android', 'web', 'mweb', 'tv_embedded'],
                    'player_skip': ['webpage', 'configs'],
                }
            },
            'cookiesfrombrowser': None,  # 쿠키 로딩 비활성화
            'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'referer': 'https://www.youtube.com/',
            'socket_timeout': 30,
            'retries': 10,
            'extract_flat': False,
            'force_generic_extractor': False,
            'geo_bypass': True,
            'geo_verification_proxy': None,
            'http_headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-us,en;q=0.5',
                'Sec-Fetch-Mode': 'navigate',
            }
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            audio_file = None
            try:
                # URL 유효성 검사
                if not url.startswith(('http://', 'https://')):
                    raise Exception("올바른 URL 형식이 아닙니다.")
                
                # URL에서 비디오 ID 추출 시도
                video_id = None
                if 'youtube.com/watch?v=' in url:
                    video_id = url.split('watch?v=')[1].split('&')[0]
                elif 'youtu.be/' in url:
                    video_id = url.split('youtu.be/')[1].split('?')[0]
                
                if video_id:
                    print(f"추출된 비디오 ID: {video_id}")
                    audio_file = f"{video_id}.mp3"
                
                # 오디오 다운로드
                print(f"비디오 정보 추출 시도: {url}")
                try:
                    info = ydl.extract_info(url, download=True)
                except Exception as e:
                    print(f"첫 번째 시도 실패: {str(e)}")
                    # 두 번째 시도: 다른 형식으로
                    ydl_opts['format'] = 'bestaudio'
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl2:
                        info = ydl2.extract_info(url, download=True)
                
                print(f"추출된 정보: {info}")
                
                if info is None:
                    raise Exception("비디오 정보를 가져올 수 없습니다.")
                
                if not video_id:
                    video_id = info.get('id')
                    if not video_id:
                        raise Exception("비디오 ID를 찾을 수 없습니다.")
                    audio_file = f"{video_id}.mp3"
                
                print(f"오디오 파일 경로: {audio_file}")
                
                if os.path.exists(audio_file):
                    # Whisper API로 STT 변환
                    with open(audio_file, "rb") as audio:
                        transcript = self.client.audio.transcriptions.create(
                            model="whisper-1",
                            file=audio,
                            language="ko"
                        )
                    os.remove(audio_file)
                    return transcript.text
                else:
                    raise Exception(f"오디오 파일을 찾을 수 없습니다. (예상 경로: {audio_file})")
            except Exception as e:
                if audio_file and os.path.exists(audio_file):
                    os.remove(audio_file)
                raise Exception(f"오디오 처리 중 오류 발생: {str(e)}")

    def create_summary(self, transcript: str) -> str:
        """자막을 요약합니다."""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "당신은 웨비나 내용을 요약하는 전문가입니다. 주어진 자막을 바탕으로 주요 내용을 요약해주세요."},
                    {"role": "user", "content": transcript}
                ]
            )
            return response.choices[0].message.content
                
        except Exception as e:
            raise Exception(f"요약 생성 중 오류 발생: {str(e)}")

    def setup_rag(self, transcript: str):
        """RAG 시스템을 설정합니다."""
        texts = self.text_splitter.split_text(transcript)
        self.vectorstore = Chroma.from_texts(texts, self.embeddings)
        
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(),
            memory=self.memory,
        )

    def ask_question(self, question: str) -> str:
        """질문에 대한 답변을 생성합니다."""
        try:
            if self.qa_chain is None:
                raise Exception("RAG 시스템이 초기화되지 않았습니다. 먼저 자막을 추출하고 요약을 생성해주세요.")
            response = self.qa_chain({"question": question})
            return response["answer"]
        except Exception as e:
            raise Exception(f"질문 답변 중 오류 발생: {str(e)}")

def initialize_session_state():
    """세션 상태를 초기화합니다."""
    if 'summarizer' not in st.session_state:
        st.session_state.summarizer = None
    if 'transcript' not in st.session_state:
        st.session_state.transcript = None
    if 'summary' not in st.session_state:
        st.session_state.summary = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'openai_api_key' not in st.session_state:
        st.session_state.openai_api_key = os.getenv("OPENAI_API_KEY")

def main():
    st.set_page_config(
        page_title="웨비나 AI 요약 및 질의응답",
        page_icon="🎥",
        layout="wide"
    )
    
    st.title("🎥 웨비나 AI 요약 및 질의응답")
    
    initialize_session_state()
    
    # 사이드바 설정
    with st.sidebar:
        st.header("설정")
        
        # OpenAI API 키 확인
        if not st.session_state.openai_api_key:
            st.error("환경변수 OPENAI_API_KEY가 설정되어 있지 않습니다.")
            return
        st.success("OpenAI API 키가 설정되어 있습니다.")
        
        # 소스 타입 선택
        source_type = st.radio(
            "입력 소스 선택",
            ["youtube", "file"],
            index=0
        )
        
        if source_type == "youtube":
            url = st.text_input("YouTube URL")
            if st.button("자막 추출", key="extract_youtube"):
                if not url:
                    st.error("YouTube URL을 입력해주세요.")
                    return
                    
                with st.spinner("자막을 추출하는 중..."):
                    try:
                        st.session_state.summarizer = WebinarSummarizer(
                            openai_api_key=st.session_state.openai_api_key
                        )
                            
                        st.session_state.transcript = st.session_state.summarizer.get_transcript(url)
                        st.success("자막 추출 완료!")
                    except Exception as e:
                        st.error(f"자막 추출 중 오류 발생: {str(e)}")
                        
        else:  # file
            uploaded_file = st.file_uploader("비디오 파일 업로드", type=["mp4", "avi", "mov"])
            if uploaded_file and st.button("자막 추출", key="extract_file"):
                with st.spinner("자막을 추출하는 중..."):
                    try:
                        # 임시 파일로 저장
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_path = tmp_file.name
                            
                        st.session_state.summarizer = WebinarSummarizer(
                            openai_api_key=st.session_state.openai_api_key
                        )
                            
                        st.session_state.transcript = st.session_state.summarizer.get_transcript(tmp_path)
                        os.remove(tmp_path)  # 임시 파일 삭제
                        st.success("자막 추출 완료!")
                    except Exception as e:
                        st.error(f"자막 추출 중 오류 발생: {str(e)}")
    
    # 메인 영역
    if st.session_state.transcript:
        # 요약 생성
        if not st.session_state.summary:
            with st.spinner("요약을 생성하는 중..."):
                try:
                    st.session_state.summary = st.session_state.summarizer.create_summary(
                        st.session_state.transcript
                    )
                    # RAG 시스템 초기화 추가
                    st.session_state.summarizer.setup_rag(st.session_state.transcript)
                    st.success("요약 및 RAG 시스템 초기화 완료!")
                except Exception as e:
                    st.error(f"요약 생성 중 오류 발생: {str(e)}")
        
        # 요약 표시
        st.header("📝 웨비나 요약")
        st.write(st.session_state.summary)
        
        # 질문-답변 영역
        st.header("💬 질문하기")
        
        # 채팅 히스토리 표시
        for msg in st.session_state.messages:
            message(msg["content"], is_user=msg["is_user"])
        
        # 질문 입력
        question = st.text_input("질문을 입력하세요", key="question_input")
        if st.button("질문하기") and question:
            with st.spinner("답변을 생성하는 중..."):
                try:
                    answer = st.session_state.summarizer.ask_question(question)
                    st.session_state.messages.append({"content": question, "is_user": True})
                    st.session_state.messages.append({"content": answer, "is_user": False})
                    st.rerun()
                except Exception as e:
                    st.error(f"답변 생성 중 오류 발생: {str(e)}")

if __name__ == "__main__":
    main()
