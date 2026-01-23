import os
from dotenv import load_dotenv
from deepgram import DeepgramClient, ClientOptionsFromEnv

# Load API Key
load_dotenv()
API_KEY = os.getenv("DEEPGRAM_API_KEY")

if not API_KEY:
    print("🚨 Error: DEEPGRAM_API_KEY not found in .env")
    exit()

def test_transcription():
    print("🚀 Testing Deepgram Connection...")
    
    try:
        # FIX: Initialize with explicit keyword argument OR empty (if using env vars)
        # Trying the most standard v3 method first:
        try:
            deepgram = DeepgramClient(api_key=API_KEY)
        except TypeError:
            # Fallback for older versions or auto-env detection
            deepgram = DeepgramClient()

        # 2. Use a URL to test
        AUDIO_URL = "https://static.deepgram.com/examples/Bueller-Life-moves-pretty-fast.wav"
        
        # 3. Simple Dictionary Options
        options = {
            "model": "nova-2",
            "smart_format": True,
        }

        print("📡 Sending data to Deepgram...")
        
        # 4. Transcribe
        source = {"url": AUDIO_URL}
        response = deepgram.listen.prerecorded.v("1").transcribe_url(source, options)
        
        # 5. Print Result
        transcript = response.results.channels[0].alternatives[0].transcript
        print("\n✅ Success! Transcript received:")
        print("-" * 30)
        print(f'"{transcript}"')
        print("-" * 30)
        return True

    except Exception as e:
        print(f"\n❌ Test Failed: {e}")
        print("\n💡 Critical Fix:")
        print("Please run this command to force the correct version:")
        print("pip uninstall -y deepgram-sdk && pip install deepgram-sdk==3.2.0")
        return False

if __name__ == "__main__":
    test_transcription()