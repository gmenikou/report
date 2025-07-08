import streamlit as st
from streamlit_webrtc import webrtc_streamer
import speech_recognition as sr
import numpy as np
import av
import threading

st.title("Real-Time Greek Voice Command Transcriber")

if "transcript" not in st.session_state:
    st.session_state.transcript = ""

recognizer = sr.Recognizer()

# This class processes the audio frames
class AudioProcessor:
    def __init__(self):
        self.audio_buffer = bytes()
        self.lock = threading.Lock()

    def recv(self, frame: av.AudioFrame):
        # Convert frame to numpy array, then to bytes
        audio = frame.to_ndarray(format="s16").tobytes()

        # Accumulate audio in buffer
        with self.lock:
            self.audio_buffer += audio

        # Process buffer every ~2 seconds (adjust as needed)
        if len(self.audio_buffer) > 32000 * 2 * 2:  # 32000 Hz, 2 bytes per sample, 2 sec
            with self.lock:
                audio_data = self.audio_buffer
                self.audio_buffer = bytes()

            # Use speech_recognition on the chunk in a thread to avoid blocking
            threading.Thread(target=process_audio, args=(audio_data,)).start()

        return frame

def process_audio(audio_bytes):
    try:
        audio_data = sr.AudioData(audio_bytes, sample_rate=32000, sample_width=2)
        text = recognizer.recognize_google(audio_data, language="el-GR").strip().lower()
        
        # Voice commands
        if text == "stop":
            lines = st.session_state.transcript.strip().split("\n")
            if lines and lines[-1]:
                line = lines[-1].strip()
                line = line[0].upper() + line[1:]
                if not line.endswith("."):
                    line += "."
                lines[-1] = line
                st.session_state.transcript = "\n".join(lines) + "\n"
            else:
                st.session_state.transcript += "\n"
        elif text == "next":
            st.session_state.transcript += "\n\n"
        else:
            st.session_state.transcript += text + " "
        
        # Force rerun to update UI
        st.experimental_rerun()
    except sr.UnknownValueError:
        pass  # ignore unrecognized audio
    except Exception as e:
        st.error(f"Error: {e}")

webrtc_streamer(
    key="audio-transcription",
    mode="sendrecv",
    audio_processor_factory=AudioProcessor,
    media_stream_constraints={"audio": True, "video": False},
    async_processing=True,
)

st.text_area("Transcript", value=st.session_state.transcript, height=300)
