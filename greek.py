import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings
import av
import queue
import numpy as np
from faster_whisper import WhisperModel

# Load Whisper with Greek language
model = WhisperModel("small", device="cpu", compute_type="int8")

# Audio queue for streaming
audio_queue = queue.Queue()

# Transcription buffer
transcript = ""
capitalize_next = True  # Track sentence beginnings

def process_commands(text):
    global capitalize_next

    text = text.strip().lower()

    if "next" in text:
        return "\n\n", True
    elif "stop" in text:
        return ". ", True
    else:
        if capitalize_next:
            text = text[:1].upper() + text[1:]
            capitalize_next = False
        return text + " ", False

def speech_worker():
    global transcript, capitalize_next
    buffer = []

    while True:
        audio_frame = audio_queue.get()
        if audio_frame is None:
            break

        pcm = audio_frame.to_ndarray().flatten().astype(np.float32) / 32768.0
        buffer.extend(pcm.tolist())

        # Process every ~3 seconds of audio
        if len(buffer) > 48000 * 3:
            segments, _ = model.transcribe(np.array(buffer), language="el", beam_size=5)

            for segment in segments:
                raw_text = segment.text.strip()
                processed, is_command = process_commands(raw_text)

                transcript += processed
                if is_command:
                    capitalize_next = True

            buffer.clear()

def audio_callback(frame: av.AudioFrame):
    audio_queue.put(frame)
    return frame

# UI
st.title("🎤 Greek Real-Time Voice Transcription")
st.markdown("Say **'stop'** to add a period. Say **'next'** for new paragraph.")

# Start WebRTC
ctx = webrtc_streamer(
    key="audio-transcription",
    mode=WebRtcMode.SENDRECV,
    client_settings=ClientSettings(
        media_stream_constraints={"audio": True, "video": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    ),
    audio_receiver_size=1024,
    audio_frame_callback=audio_callback,
    async_processing=True,
)

# Start transcribing in background
if ctx.state.playing:
    st.info("Listening... Speak Greek clearly into your USB mic.")
    threading = st.session_state.get("threading")

    if not threading:
        import threading as th
        t = th.Thread(target=speech_worker, daemon=True)
        t.start()
        st.session_state["threading"] = True

# Output
st.text_area("📄 Transcript", transcript, height=300)
