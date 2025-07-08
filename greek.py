import streamlit as st
import whisper
import tempfile
import os
from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings
import av

# Load a smaller Whisper model due to memory limits on Streamlit Cloud
model = whisper.load_model("small")  # use "large" locally only

st.title("🎙️ Greek Voice Transcriber with Whisper")

st.markdown("This transcribes your speech using OpenAI's Whisper model. Say 'stop' or 'next' for commands.")

# Setup audio receiver
class AudioProcessor:
    def __init__(self):
        self.audio_buffer = b""

    def recv(self, frame: av.AudioFrame):
        self.audio_buffer += frame.to_ndarray().tobytes()
        return av.AudioFrame.from_ndarray(frame.to_ndarray(), layout=frame.layout.name)

audio_processor = AudioProcessor()

webrtc_streamer(
    key="audio-transcription",
    mode=WebRtcMode.SENDONLY,
    client_settings=ClientSettings(
        media_stream_constraints={"audio": True, "video": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    ),
    audio_receiver_factory=lambda: audio_processor,
)

if st.button("🔊 Transcribe Audio"):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        tmpfile.write(audio_processor.audio_buffer)
        tmpfile_path = tmpfile.name

    # Transcribe using whisper
    result = model.transcribe(tmpfile_path, language="el")  # 'el' for Greek
    text = result["text"]

    # Handle voice commands
    text = text.strip()
    if "stop" in text.lower():
        text = text.replace("stop", "").strip().capitalize() + "."
    if "next" in text.lower():
        text = text.replace("next", "").strip() + "\n\n"

    st.success("Transcription:")
    st.write(text)

    os.remove(tmpfile_path)
