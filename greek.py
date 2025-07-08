import streamlit as st
from streamlit_webrtc import webrtc_streamer
import whisper
import av
import numpy as np
import queue
import threading
import re

st.title("Streamlit Whisper Greek Transcriber with Voice Commands")

# Load Whisper model once (use smaller model for faster performance)
@st.cache_resource(show_spinner=True)
def load_model():
    return whisper.load_model("base")  # you can switch to "large" if resources allow

model = load_model()

audio_queue = queue.Queue()

def audio_frame_callback(frame: av.AudioFrame):
    audio = frame.to_ndarray(format="flt32").flatten()
    audio_queue.put(audio)
    return frame

if "transcript" not in st.session_state:
    st.session_state["transcript"] = ""

if "buffer" not in st.session_state:
    st.session_state["buffer"] = ""

def apply_voice_commands(text, current_transcript):
    text_lower = text.lower().strip()
    if text_lower == "stop":
        # Capitalize last sentence first letter and add period if missing
        sentences = re.split(r'([.?!])', current_transcript.strip())
        if len(sentences) >= 2:
            # sentences list alternates text and punctuation, e.g. ['Hello', '.', ' world', '.']
            # find last non-empty sentence text:
            last_text = sentences[-2].strip()
            if last_text:
                last_text = last_text[0].upper() + last_text[1:]
                sentences[-2] = last_text
                # Add period if last punctuation missing or not period
                if sentences[-1] not in ['.', '?', '!']:
                    sentences.append('.')
                elif sentences[-1] != '.':
                    sentences[-1] = '.'
            new_text = "".join(sentences)
        else:
            # If no sentence detected, just capitalize all and add period
            new_text = current_transcript.strip().capitalize() + "."
        return new_text, True  # True means command handled

    elif text_lower == "next":
        # Add paragraph break
        if current_transcript and not current_transcript.endswith("\n\n"):
            current_transcript += "\n\n"
        return current_transcript, True

    else:
        return current_transcript + " " + text, False

def transcribe_loop():
    buffer = np.zeros((0,), dtype=np.float32)
    while True:
        audio_chunk = audio_queue.get()
        buffer = np.concatenate((buffer, audio_chunk))
        # Process approx 2 seconds of audio (16000Hz * 2s)
        if len(buffer) >= 32000:
            segment = buffer[:32000]
            buffer = buffer[32000:]
            # Whisper expects float32 numpy audio in range [-1, 1]
            result = model.transcribe(segment, language="el", fp16=False)
            text = result["text"].strip()
            if text:
                new_transcript, handled = apply_voice_commands(text, st.session_state["transcript"])
                st.session_state["transcript"] = new_transcript
                st.experimental_rerun()

webrtc_streamer(
    key="audio-transcriber",
    audio_frame_callback=audio_frame_callback,
    media_stream_constraints={"audio": True, "video": False},
    async_processing=True,
)

st.text_area("Transcription Output", value=st.session_state["transcript"], height=300)

if "thread" not in st.session_state:
    threading.Thread(target=transcribe_loop, daemon=True).start()
    st.session_state["thread"] = True
