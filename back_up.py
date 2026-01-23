import streamlit as st
import os
import asyncio
import edge_tts
from io import BytesIO
from dotenv import load_dotenv

# --- IMPORTS FOR BROWSER RECORDING ---
from streamlit_mic_recorder import mic_recorder
import speech_recognition as sr

from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Virtuans Voice Agent")
load_dotenv()

# Verify Keys
if not os.getenv("OPENROUTER_API_KEY") or not os.getenv("GOOGLE_API_KEY"):
    st.error("🚨 Missing API Keys! Check your .env file.")
    st.stop()

# --- AUDIO INPUT UTILS (Browser -> Python) ---
def speech_to_text(audio_bytes):
    """Converts audio bytes to text using Google Speech Recognition."""
    r = sr.Recognizer()
    try:
        audio_file = sr.AudioFile(BytesIO(audio_bytes))
        with audio_file as source:
            # Clean up noise
            r.adjust_for_ambient_noise(source)
            audio_data = r.record(source)
            text = r.recognize_google(audio_data)
            return text
    except sr.UnknownValueError:
        st.warning("Could not understand audio.")
        return None
    except sr.RequestError as e:
        st.error(f"Could not request results; {e}")
        return None
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return None

# --- AUDIO OUTPUT UTILS (EdgeTTS - Free & High Quality) ---
def text_to_speech(text, model_name):
    """Converts text to audio using Edge-TTS (No API Key needed)."""
    if not text:
        return None
    
    # Assign specific voices to specific agents
    voice = "en-US-AriaNeural"  # Default
    if "Gemini" in model_name:
        voice = "en-US-AvaNeural"       # Female
    elif "DeepSeek" in model_name:
        voice = "en-US-ChristopherNeural" # Male, Deep
    elif "Kimi" in model_name:
        voice = "en-US-EricNeural"      # Male, Energetic

    output_file = f"temp_{model_name}.mp3"
    
    async def generate():
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(output_file)

    try:
        # Create a safe async loop for Streamlit
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(generate())
        loop.close()
        
        # Read file into bytes
        with open(output_file, "rb") as f:
            audio_bytes = BytesIO(f.read())
        
        # Cleanup
        if os.path.exists(output_file):
            os.remove(output_file)
            
        return audio_bytes
        
    except Exception as e:
        print(f"TTS Error for {model_name}: {e}")
        return None

# --- RAG PIPELINE ---
@st.cache_resource
def setup_rag_chain():
    # 1. Load Vector Store
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", max_retries=5)
    
    try:
        vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    except Exception as e:
        st.error(f"Error loading Vector DB: {e}. Run ingest.py first!")
        return None

    # 2. Prompts
    template = """You are a helpful assistant for Sunmarke School.
    Answer the user's question based ONLY on the following context. 
    If unsure, say you don't know. Keep it under 3 sentences for voice output.

    Context: {context}
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # 3. Initialize Models (Fixed IDs + Max Tokens)
    openrouter_base = "https://openrouter.ai/api/v1"
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    headers = {"HTTP-Referer": "http://localhost:8501", "X-Title": "Virtuans Agent"}

    # --- UPDATED MODEL IDs HERE ---
    llm_gemini = ChatOpenAI(
        api_key=openrouter_key, 
        model="google/gemini-2.5-flash", # Updated Free Model ID
        # model="google/gemini-2.5-pro-exp-03-25:free", # Updated Free Model ID
        base_url=openrouter_base, 
        default_headers=headers,
        max_tokens=500
        
    )
    
    llm_deepseek = ChatOpenAI(
        model="deepseek/deepseek-chat", # Standard DeepSeek ID
        api_key=openrouter_key, 
        base_url=openrouter_base, 
        default_headers=headers,
        max_tokens=500
    )
    
    llm_kimi = ChatOpenAI(
        model="moonshotai/kimi-k2-0905", # Updated Vendor Prefix
        api_key=openrouter_key, 
        base_url=openrouter_base, 
        default_headers=headers,
        max_tokens=500
    )


    # 4. Chains
    rag_setup = {"context": retriever | (lambda d: "\n\n".join([x.page_content for x in d])), "question": RunnablePassthrough()}

    return {
        "gemini": rag_setup | prompt | llm_gemini | StrOutputParser(),
        "deepseek": rag_setup | prompt | llm_deepseek | StrOutputParser(),
        "kimi": rag_setup | prompt | llm_kimi | StrOutputParser()
    }

# --- MAIN UI ---
def main():
    st.title("🤖 Virtuans Voice Agent")
    
    chains = setup_rag_chain()
    if not chains: st.stop()

    col_input, col_text = st.columns([1, 4])
    
    user_query = None

    # --- INPUT: VOICE ---
    with col_input:
        st.write("Click to Speak:")
        # This widget records audio in the browser
        audio = mic_recorder(
            start_prompt="🎤 Record",
            stop_prompt="⏹️ Stop",
            key='recorder',
            just_once=False,
            use_container_width=True
        )
        
        if audio:
            with st.spinner("Transcribing..."):
                user_query = speech_to_text(audio['bytes'])

    # --- INPUT: TEXT FALLBACK ---
    with col_text:
        text_input = st.text_input("Or type here:", value=user_query if user_query else "")
        if text_input:
            user_query = text_input

    # --- PROCESSING ---
    if user_query:
        st.divider()
        st.subheader(f"Q: {user_query}")
        
        col1, col2, col3 = st.columns(3)
        with col1: status_g = st.empty(); status_g.info("Gemini thinking...")
        with col2: status_k = st.empty(); status_k.info("Kimi thinking...")
        with col3: status_d = st.empty(); status_d.info("DeepSeek thinking...")

        async def get_responses(q):
            return await asyncio.gather(
                chains["gemini"].ainvoke(q),
                chains["kimi"].ainvoke(q),
                chains["deepseek"].ainvoke(q),
                return_exceptions=True
            )

        try:
            # Main LLM Execution
            results = asyncio.run(get_responses(user_query))
            resp_g, resp_k, resp_d = results
            
            for s in [status_g, status_k, status_d]: s.empty()

            # Helper to render column
            def display_result(col, name, resp):
                with col:
                    st.markdown(f"**{name}**")
                    if isinstance(resp, Exception):
                        st.error(f"API Error: {str(resp)}")
                    else:
                        st.success(resp)
                        # Generate Audio
                        audio_fp = text_to_speech(resp, name)
                        if audio_fp: 
                            st.audio(audio_fp, format="audio/mp3")
                        else:
                            st.warning("Audio unavailable")

            display_result(col1, "Gemini", resp_g)
            display_result(col2, "Kimi", resp_k)
            display_result(col3, "DeepSeek", resp_d)

        except Exception as e:
            st.error(f"System Error: {e}")

if __name__ == "__main__":
    main()