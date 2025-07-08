import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import whisper
import numpy as np
import av
import tempfile
import re

# Load Whisper model once (cached for performance)
@st.cache_resource
def load_model():
    return whisper.load_model("large")

model = load_model()

st.title("Live Whisper Large Transcription with Voice Commands")

# Store transcript state
if "transcript" not in st.session_state:
    st.session_state.transcript = ""

def process_commands(text):
    """
    Detect voice commands and apply formatting:
    - "stop": capitalize first letter and add period if not present
    - "next": insert paragraph break (two newlines)
    Return cleaned text and updated transcript string.
    """
    words = text.strip().lower().split()

    if not words:
        return "", ""

    # Commands only if single word or last word
    last_word = words[-1]

    if last_word == "stop":
        # Remove "stop" from transcription
        text = text.rsplit(" ", 1)[0].strip()

        # Capitalize first letter of last sentence and add period if missing
        if st.session_state.transcript:
            # Capitalize first letter of last sentence in transcript if needed
            sentences = re.split(r'([.!?])', st.session_state.transcript.strip())
            if sentences:
                # Combine last sentence with punctuation
                last_sentence = "".join(sentences[-2:]).strip()
                # Capitalize first char
                last_sentence = last_sentence.capitalize()
                if not last_sentence.endswith('.'):
                    last_sentence += '.'
                # Replace last sentence in transcript
                st.session_state.transcript = "".join(sentences[:-2]) + last_sentence + " "

        return "", ""  # Don't add the command word itself to transcript

    elif last_word == "next":
        # Remove "next" from transcription
        text = text.rsplit(" ", 1)[0].strip()

        # Insert paragraph break
        st.session_state.transcript += "\n\n"

        return "", ""  # Don't add "next" to transcript

    else:
        return text, text + " "

def audio_frame_callback(frame: av.AudioFrame):
    # Convert audio frame to numpy array
    audio = frame.to_ndarray(format="flt32", layout="mono")

    # Resample from 48000 to 16000 Hz (Whisper requirement)
    audio_16k = whisper.audio.resample(audio.flatten(), 48000, 16000)

    # Save temp wav file to transcribe
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        import soundfile as sf
        sf.write(tmp.name, audio_16k, 16000)

        # Run Whisper transcription with Greek language (change if needed)
        result = model.transcribe(tmp.name, language="el")
        text = result["text"].strip()

    if text:
        cleaned_text, appended_text = process_commands(text)
        if appended_text:
            st.session_state.transcript += appended_text

    # Update transcript display in Streamlit (async-safe)
    st.session_state.transcript = st.session_state.transcript.strip()
    st.session_state.transcript_display = st.session_state.transcript.replace("\n", "\n")

    return av.AudioFrame.from_ndarray(audio, layout="mono")

# UI
st.text_area(
    "Transcription",
    value=st.session_state.get("transcript_display", ""),
    height=300,
    key="text_area",
    label_visibility="visible"
)

# Start WebRTC streamer for audio input, no video
webrtc_streamer(
    key="whisper-large-voicecmd",
    mode=WebRtcMode.SENDRECV,
    audio_frame_callback=audio_frame_callback,
    media_stream_constraints={"audio": True, "video": False},
    async_processing=True,
)
