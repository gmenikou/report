import streamlit as st
import whisper
import sounddevice as sd
import numpy as np
import queue
import threading

st.title("Live Whisper Transcription with Voice Commands")

# Load model once (not cached)
@st.cache_resource(show_spinner=False)
def load_model():
    return whisper.load_model("base")  # use "small" or "tiny" for faster

model = load_model()

# Audio recording params
samplerate = 16000  # Whisper needs 16kHz audio
channels = 1
blocksize = 1024

audio_q = queue.Queue()

# Buffer to hold audio chunks
audio_buffer = []

def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    audio_q.put(indata.copy())

# Start recording stream in a background thread
def start_recording():
    stream = sd.InputStream(
        samplerate=samplerate,
        channels=channels,
        callback=audio_callback,
        blocksize=blocksize,
    )
    stream.start()
    return stream

# Start stream globally once
if "stream" not in st.session_state:
    st.session_state.stream = start_recording()

# Text area for transcription output
if "transcript" not in st.session_state:
    st.session_state.transcript = ""

st.text_area("Transcription", value=st.session_state.transcript, height=300, key="transcript_area")

# Processing thread function
def process_audio():
    while True:
        try:
            data = audio_q.get(timeout=1)
            audio_buffer.append(data.flatten())
            
            # Process every ~2 seconds of audio
            if len(audio_buffer)*blocksize >= samplerate*2:
                audio_np = np.concatenate(audio_buffer)
                audio_buffer.clear()
                
                # Whisper expects float32 np array normalized between -1 and 1
                audio_float = audio_np.astype(np.float32)
                audio_float = audio_float / np.max(np.abs(audio_float))
                
                # Transcribe
                result = model.transcribe(audio_float, language="el", fp16=False)
                text = result["text"].strip()
                if not text:
                    continue
                
                # Handle voice commands
                text_lower = text.lower()
                if "stop" in text_lower:
                    # Capitalize first letter and add period
                    if st.session_state.transcript:
                        st.session_state.transcript = st.session_state.transcript.rstrip() + ". "
                    # Capitalize next word (simulate new sentence)
                    text = text_lower.replace("stop", "").strip().capitalize()
                elif "next" in text_lower:
                    # Insert new paragraph
                    st.session_state.transcript += "\n\n"
                    text = text_lower.replace("next", "").strip()
                
                # Append recognized text
                if text:
                    if st.session_state.transcript and not st.session_state.transcript.endswith(("\n", " ")):
                        st.session_state.transcript += " "
                    st.session_state.transcript += text
                
                # Update text area in main thread
                st.experimental_rerun()
        except queue.Empty:
            continue

if "thread" not in st.session_state:
    thread = threading.Thread(target=process_audio, daemon=True)
    thread.start()
    st.session_state.thread = thread
