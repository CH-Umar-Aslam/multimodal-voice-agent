import streamlit as st
import os, time, asyncio, edge_tts
from io import BytesIO
from dotenv import load_dotenv

# --- IMPORTS FOR BROWSER RECORDING ---
from streamlit_mic_recorder import mic_recorder
from pydub import AudioSegment  # NEW: For converting WebM -> WAV
from deepgram import DeepgramClient
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

# --- AUDIO INPUT
def speech_to_text(audio_bytes):
    """
    Sends audio bytes directly to Deepgram using SDK v3.
    """
    try:
        # Initialize Deepgram Client
        # FIX: We explicitly pass 'api_key=' to avoid the __init__ error
        deepgram_key = os.getenv("DEEPGRAM_API_KEY")
        deepgram = DeepgramClient(api_key=deepgram_key)

        # Configure Options
        # FIX: Using a simple dictionary is safer and works on all versions
        options = {
            "model": "nova-2",
            "smart_format": True,
            "language": "en-US",
        }

        # Send Payload
        # Deepgram expects a dictionary payload for buffers
        payload = {"buffer": audio_bytes}
        
        # Transcribe
        response = deepgram.listen.prerecorded.v("1").transcribe_file(payload, options)
        
        # Extract Transcript
        transcript = response.results.channels[0].alternatives[0].transcript
        return transcript

    except Exception as e:
        st.error(f"Deepgram STT Error: {e}")
        return None

# --- AUDIO OUTPUT UTILS (EdgeTTS) ---
def text_to_speech(text, model_name):
    """Converts text to audio using Edge-TTS."""
    if not text:
        return None
    
    # Assign specific voices
    voice = "en-US-AriaNeural"
    if "Gemini" in model_name:
        voice = "en-US-AvaNeural"
    elif "DeepSeek" in model_name:
        voice = "en-US-ChristopherNeural"
    elif "Kimi" in model_name:
        voice = "en-US-EricNeural"

    output_file = f"temp_{model_name}.mp3"
    
    async def generate():
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(output_file)

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(generate())
        loop.close()
        
        with open(output_file, "rb") as f:
            audio_bytes = BytesIO(f.read())
        
        if os.path.exists(output_file):
            os.remove(output_file)
            
        return audio_bytes
        
    except Exception as e:
        print(f"TTS Error for {model_name}: {e}")
        return None

# --- RAG PIPELINE ---
@st.cache_resource
def setup_rag_chain():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", max_retries=5)
    
    try:
        vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    except Exception as e:
        st.error(f"Error loading Vector DB: {e}. Run ingest.py first!")
        return None

    template = """You are a specialized AI Assistant for Sunmarke School.

        TASK:
        1. If the answer is in the context, provide a concise and short response  (max 3 sentences).
        2. If the question is NOT about Sunmarke School or the information is missing, you must politely shortly introduce your specialization.

        DYNAMIC FALLBACK RULE:
        - Do NOT use a fixed or hardcoded response for out-of-scope questions.
        - Instead, in your own unique personality and style, precise that you are a dedicated Sunmarke School assistant and can only help with school-related matters (like admissions, fees, or curriculum).
        - Be creative but professional in how you phrase your limitation. 
        - Mention the user's topic to show you understood it, then pivot back to your role at Sunmarke.

        Context: {context}
        Question: {question}

        Response:"""
        
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
        temperature=0,
        default_headers=headers,
        max_tokens=500
        
    )
    
    llm_deepseek = ChatOpenAI(
        model="deepseek/deepseek-chat", # Standard DeepSeek ID
        api_key=openrouter_key, 
        base_url=openrouter_base, 
        temperature=0,
        default_headers=headers,
        max_tokens=500
    )
    
    llm_kimi = ChatOpenAI(
        model="moonshotai/kimi-k2-0905", # Updated Vendor Prefix
        api_key=openrouter_key, 
        base_url=openrouter_base,
        temperature=0, 
        default_headers=headers,
        max_tokens=500
    )

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
        start_time = time.perf_counter()
        st.divider()
        
        st.subheader(f"Q: {user_query}")
        
        col1, col2, col3 = st.columns(3)
        with col1: status_g = st.empty(); status_g.info("Gemini thinking...")
        with col2: status_k = st.empty(); status_k.info("Kimi thinking...")
        with col3: status_d = st.empty(); status_d.info("DeepSeek thinking...")

        async def get_model_response(name, chain, q):
            start = time.perf_counter()
            try:
                response = await chain.ainvoke(q)
                duration = time.perf_counter() - start
                return response, duration
            except Exception as e:
                return e, 0

        async def get_responses(q):
            return await asyncio.gather(
                get_model_response("gemini", chains["gemini"], q),
                get_model_response("kimi", chains["kimi"], q),
                get_model_response("deepseek", chains["deepseek"], q),
                return_exceptions=True
            )

        try:
            results = asyncio.run(get_responses(user_query))
            total_duration = time.perf_counter() - start_time
            resp_g, resp_k, resp_d = results
            
            for s in [status_g, status_k, status_d]: s.empty()

            def display_result(col, name, resp_data):
                # resp_data mein ab (response, duration) ka tuple hoga
                resp, duration = resp_data

                with col:
                    st.markdown(f"### {name}")
                    if isinstance(resp, Exception):
                        st.error(f"API Error: {str(resp)}")
                    else:
                        st.success(resp)
                       
                        
                        # Buttons text ke liye hain, isliye audio se pehle dikha dein
                        c1, c2 ,c3= st.columns(3)
                        with c1: st.caption("👍")
                        with c2: st.caption("👎")
                        with c3: st.caption(f"⏱ {duration:.2f}s")

                        
                        # Audio output
                        audio_fp = text_to_speech(resp, name)
                        if audio_fp: 
                            st.audio(audio_fp, format="audio/mp3")
                        else:
                            st.warning("Audio unavailable")

            # Calling part (In main loop)
            results = asyncio.run(get_responses(user_query))
            resp_g_data, resp_k_data, resp_d_data = results

            display_result(col1, "Gemini", resp_g_data)
            display_result(col2, "Kimi", resp_k_data)
            display_result(col3, "DeepSeek", resp_d_data)
          

        except Exception as e:
            st.error(f"System Error: {e}")

if __name__ == "__main__":
    main()