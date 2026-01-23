import streamlit as st
import os
import asyncio
import speech_recognition as sr
from gtts import gTTS
from io import BytesIO
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Virtuans Voice Agent")
load_dotenv()

# Verify Keys Exist
if not os.getenv("OPENROUTER_API_KEY") or not os.getenv("GOOGLE_API_KEY"):
    st.error("🚨 Missing API Keys! Please ensure OPENROUTER_API_KEY and GOOGLE_API_KEY are set in your .env file.")
    st.stop()

# --- AUDIO UTILS ---
def record_audio():
    """Records audio from the microphone and returns the text."""
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎤 Listening... Speak now!")
        try:
            # Adjust ambient noise for better accuracy
            r.adjust_for_ambient_noise(source, duration=1)
            audio = r.listen(source, timeout=5, phrase_time_limit=10)
            st.success("Processing audio...")
            text = r.recognize_google(audio)
            return text
        except sr.WaitTimeoutError:
            st.warning("No speech detected.")
            return None
        except sr.UnknownValueError:
            st.warning("Could not understand audio.")
            return None
        except Exception as e:
            st.error(f"Microphone error: {e}")
            return None

def text_to_speech(text, model_name):
    """Converts text to audio using gTTS."""
    try:
        mp3_fp = BytesIO()
        # English, slow=False
        tts = gTTS(text=text, lang='en', slow=False)
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        return mp3_fp
    except Exception as e:
        st.error(f"TTS Error for {model_name}: {e}")
        return None

# --- RAG PIPELINE SETUP ---
@st.cache_resource
def setup_rag_chain():
    """Initializes Vector DB and OpenRouter LLM Chains."""
    
    # 1. Load Vector Store
    # MUST match the model used in ingest.py exactly
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        max_retries=5
    )
    
    try:
        vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    except Exception as e:
        st.error(f"Error loading Vector DB: {e}. Please run ingest.py first!")
        return None

    # 2. Define Prompts
    template = """You are a helpful assistant for Sunmarke School.
    Answer the user's question based ONLY on the following context. 
    If the answer is not in the context, politely say you don't have that information.
    Keep your answer concise (under 3 sentences) suitable for voice output.

    Context:
    {context}

    Question:
    {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # 3. Initialize LLMs via OpenRouter
    # Base configuration for OpenRouter
    openrouter_base = "https://openrouter.ai/api/v1"
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    
    # Common headers for OpenRouter (optional but recommended)
    headers = {
        "HTTP-Referer": "https://localhost:8501", # Site URL
        "X-Title": "Virtuans Voice Agent",       # App Name
    }

    # -- Model 1: Gemini (via OpenRouter) --
    llm_gemini = ChatOpenAI(
        model="google/gemini-pro",  # OpenRouter Slug
        api_key=openrouter_key,
        base_url=openrouter_base,
        temperature=0.3,
        default_headers=headers
    )
    
    # -- Model 2: DeepSeek (via OpenRouter) --
    llm_deepseek = ChatOpenAI(
        model="deepseek/deepseek-chat", # OpenRouter Slug
        api_key=openrouter_key,
        base_url=openrouter_base,
        temperature=0.3,
        default_headers=headers
    )

    # -- Model 3: Kimi / Moonshot (via OpenRouter) --
    llm_kimi = ChatOpenAI(
        model="moonshot/moonshot-v1-8k", # OpenRouter Slug
        api_key=openrouter_key,
        base_url=openrouter_base,
        temperature=0.3,
        default_headers=headers
    )

    # 4. Create Chains
    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

    # Shared RAG context builder
    rag_setup = {"context": retriever | format_docs, "question": RunnablePassthrough()}

    chain_gemini = rag_setup | prompt | llm_gemini | StrOutputParser()
    chain_deepseek = rag_setup | prompt | llm_deepseek | StrOutputParser()
    chain_kimi = rag_setup | prompt | llm_kimi | StrOutputParser()

    return {
        "gemini": chain_gemini,
        "deepseek": chain_deepseek,
        "kimi": chain_kimi
    }

# --- MAIN UI ---
def main():
    st.title("🤖 Virtuans Voice Agent Challenge")
    st.markdown("Ask questions about **Sunmarke School**. Powered by **OpenRouter** (Gemini, DeepSeek, Kimi).")
    
    # Initialize RAG
    chains = setup_rag_chain()
    if not chains:
        st.stop()

    # --- INPUT SECTION ---
    col_input_1, col_input_2 = st.columns([1, 4])
    
    user_query = None
    
    with col_input_1:
        # Real-time Recording Button
        if st.button("🎤 Hold to Record", type="primary"):
            user_query = record_audio()

    # Debug/Fallback Text Input
    with col_input_2:
        text_input = st.text_input("Or type your question here:", key="text_input")
        if text_input and not user_query:
            user_query = text_input

    # --- PROCESSING ---
    if user_query:
        st.divider()
        st.markdown(f"### 🗣️ Question: *{user_query}*")
        
        # UI Placeholders for Loading State
        col1, col2, col3 = st.columns(3)
        with col1: 
            status_g = st.empty()
            status_g.info("Gemini thinking...")
        with col2: 
            status_k = st.empty()
            status_k.info("Kimi thinking...")
        with col3: 
            status_d = st.empty()
            status_d.info("DeepSeek thinking...")

        # Async Function for Parallel Execution
        async def get_responses(query):
            # Run all three chains in parallel
            results = await asyncio.gather(
                chains["gemini"].ainvoke(query),
                chains["kimi"].ainvoke(query),
                chains["deepseek"].ainvoke(query),
                return_exceptions=True # Prevent one failure from crashing all
            )
            return results

        try:
            # Run Async Loop
            results = asyncio.run(get_responses(user_query))
            resp_gemini, resp_kimi, resp_deepseek = results
            
            # Clear loading status
            status_g.empty()
            status_k.empty()
            status_d.empty()

        except Exception as e:
            st.error(f"Error during LLM inference: {e}")
            st.stop()

        # --- DISPLAY RESULTS ---
        
        # Column 1: Gemini
        with col1:
            st.subheader("Gemini")
            if isinstance(resp_gemini, Exception):
                st.error(f"Error: {resp_gemini}")
            else:
                st.success("Response Ready")
                st.write(resp_gemini)
                audio_g = text_to_speech(resp_gemini, "Gemini")
                if audio_g: st.audio(audio_g, format="audio/mp3")

        # Column 2: Kimi
        with col2:
            st.subheader("Kimi (Moonshot)")
            if isinstance(resp_kimi, Exception):
                st.error(f"Error: {resp_kimi}")
            else:
                st.success("Response Ready")
                st.write(resp_kimi)
                audio_k = text_to_speech(resp_kimi, "Kimi")
                if audio_k: st.audio(audio_k, format="audio/mp3")

        # Column 3: DeepSeek
        with col3:
            st.subheader("DeepSeek")
            if isinstance(resp_deepseek, Exception):
                st.error(f"Error: {resp_deepseek}")
            else:
                st.success("Response Ready")
                st.write(resp_deepseek)
                audio_d = text_to_speech(resp_deepseek, "DeepSeek")
                if audio_d: st.audio(audio_d, format="audio/mp3")

if __name__ == "__main__":
    main()