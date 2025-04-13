import sounddevice as sd
import numpy as np
import requests
import threading
import time
import io
import scipy.io.wavfile as wavfile
import queue

# --- Configuration ---
SAMPLE_RATE = 16000  # Sample rate expected by Whisper
CHANNELS = 1         # Mono audio
DTYPE = np.int16     # Data type for audio samples (check whisper.cpp server needs)
                     # float32 is also common for processing, wav needs int16 usually

CHUNK_DURATION_S = 6 # Duration of each audio chunk in seconds
OVERLAP_DURATION_S = 1  # Duration of overlap in seconds
BUFFER_DURATION_S = CHUNK_DURATION_S # Minimum buffer size needed to extract a chunk
TARGET_INPUT_DEVICE_NAME = "CABLE Output (VB-Audio Virtual Cable)" # Or part of the name


# --- Helper Function to Find Device --- drivers https://vb-audio.com/Cable/index.htm
def find_device_id(name_part, kind):
    """Finds a device ID based on a partial name and kind ('input' or 'output')."""
    devices = sd.query_devices()
    print("\nAvailable Audio Devices:")
    print(devices) # Print all devices to help debugging
    print("-" * 30)
    for i, device in enumerate(devices):
        # Check if the partial name is in the device name and if it supports input channels
        if name_part.lower() in device['name'].lower() and device[f'max_{kind}_channels'] > 0:
            print(f"Found matching {kind} device: ID {i}, Name: {device['name']}")
            return i
    print(f"Warning: Could not find an {kind} device containing the name '{name_part}'. Using default.")
    return None # Use default if not found


# Calculate samples
CHUNK_SAMPLES = CHUNK_DURATION_S * SAMPLE_RATE
OVERLAP_SAMPLES = OVERLAP_DURATION_S * SAMPLE_RATE
BUFFER_SAMPLES = BUFFER_DURATION_S * SAMPLE_RATE
# Amount to discard from the buffer after processing a chunk
DISCARD_SAMPLES = CHUNK_SAMPLES - OVERLAP_SAMPLES

# Whisper.cpp server endpoint URL (replace with your actual server URL)
WHISPER_API_URL = "http://localhost:8080/inference" # Common default, check yours

# --- Global Variables ---
audio_buffer = np.array([], dtype=DTYPE)
buffer_lock = threading.Lock()
stop_event = threading.Event()
transcription_queue = queue.Queue() # To pass transcriptions back to main thread safely

# --- Audio Callback ---
def audio_callback(indata, frames, time, status):
    """This is called (from a separate thread) for each block of audio data."""
    if status:
        print(f"Audio callback status: {status}", flush=True)

    global audio_buffer
    with buffer_lock:
        # Append new data, ensuring it stays within reasonable bounds if processing falls behind
        # Although the discard logic should handle this, adding a max size is safer.
        # MAX_BUFFER_SAMPLES = BUFFER_SAMPLES * 3 # Example: max 3 chunks worth
        audio_buffer = np.append(audio_buffer, indata[:, 0]) # Assuming mono, take first channel
        # Trim buffer if it gets excessively large (optional safety)
        # if len(audio_buffer) > MAX_BUFFER_SAMPLES:
        #     audio_buffer = audio_buffer[-MAX_BUFFER_SAMPLES:]


# --- Processing Thread ---
def processing_thread_func():
    """Continuously processes audio chunks from the buffer."""
    global audio_buffer
    print("Processing thread started.", flush=True)

    while not stop_event.is_set():
        chunk_to_process = None
        current_buffer_len = 0

        with buffer_lock:
            current_buffer_len = len(audio_buffer)
            if current_buffer_len >= CHUNK_SAMPLES:
                # Extract the latest chunk
                chunk_to_process = audio_buffer[-CHUNK_SAMPLES:].copy()
                # Discard the older non-overlapping part
                audio_buffer = audio_buffer[DISCARD_SAMPLES:]
                # print(f"Buffer: {current_buffer_len} -> {len(audio_buffer)} samples", flush=True) # Debug

        if chunk_to_process is not None:
            print(f"Processing chunk of {len(chunk_to_process)} samples...", flush=True)
            try:
                # 1. Convert NumPy array to WAV format in memory
                wav_bytes_io = io.BytesIO()
                wavfile.write(wav_bytes_io, SAMPLE_RATE, chunk_to_process.astype(np.int16)) # Ensure int16 for WAV
                wav_bytes_io.seek(0) # Rewind the buffer to the beginning

                # 2. Prepare multipart/form-data
                files = {'file': ('audio.wav', wav_bytes_io, 'audio/wav')}

                # Optional: Add parameters if your whisper.cpp server supports them
                # data = {
                #     'temperature': '0.0',
                #     'response-format': 'json' # or 'text'
                # }
                # response = requests.post(WHISPER_API_URL, files=files, data=data, timeout=10)

                # 3. Send POST request
                start_time = time.time()
                response = requests.post(WHISPER_API_URL, files=files, timeout=60) # Increased timeout
                end_time = time.time()


                response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

                # 4. Process response (assuming JSON like {'text': '...'})
                # Adjust parsing based on your server's actual response format
                result = response.json()
                transcription = result.get('text', '').strip()

                if transcription:
                    processing_time = end_time - start_time
                    print(f"API Time: {processing_time:.2f}s | Transcription: {transcription}", flush=True)
                    transcription_queue.put(transcription) # Put result in queue
                else:
                    print(f"API Time: {processing_time:.2f}s | (No transcription)", flush=True)


            except requests.exceptions.RequestException as e:
                print(f"API Error: {e}", flush=True)
            except Exception as e:
                print(f"Error processing chunk: {e}", flush=True)
            finally:
                 # Clean up the BytesIO object
                 wav_bytes_io.close()

            # Optional small delay if needed, but processing/API call is the main delay
            # time.sleep(0.05)

        else:
            # Wait a bit if buffer doesn't have enough data
            time.sleep(0.1)

    print("Processing thread finished.", flush=True)

# --- Main Execution ---
if __name__ == "__main__":
    print("Starting Real-time ASR Application...")
    # ... (print other config)
    print("Starting Real-time ASR Application...")
    print(f"Sample Rate: {SAMPLE_RATE} Hz")
    print(f"Chunk Duration: {CHUNK_DURATION_S}s")
    print(f"Overlap Duration: {OVERLAP_DURATION_S}s")
    print(f"API Endpoint: {WHISPER_API_URL}")
    print("-" * 30)

    # Find the specific input device ID
    input_device_id = find_device_id(TARGET_INPUT_DEVICE_NAME, 'input')  

    # Start the processing thread
    processor_thread = threading.Thread(target=processing_thread_func)
    processor_thread.start()

    # Start the audio stream
    try:
        print(f"Attempting to use input device ID: {input_device_id if input_device_id is not None else 'Default'}")
        print("Starting audio stream...")
        with sd.InputStream(samplerate=SAMPLE_RATE,
                            channels=CHANNELS,
                            dtype=DTYPE,
                            callback=audio_callback,
                            blocksize=int(SAMPLE_RATE * 0.5),
                            device=input_device_id): # <--- Specify the device ID here
            print("Audio stream active capturing system audio (via VB-CABLE). Press Ctrl+C to stop.")
            # Keep the main thread alive, optionally process transcriptions from queue
            while True:
                try:
                    # Get transcriptions from the queue (non-blocking)
                    transcription = transcription_queue.get_nowait()
                    # You could do something more sophisticated here, like concatenating results
                    # print(f"Main Thread Received: {transcription}")
                except queue.Empty:
                    pass # No new transcription yet
                except KeyboardInterrupt: # Allow Ctrl+C to break the loop
                    print("\nCtrl+C received, stopping...")
                    break
                time.sleep(0.1) # Prevent busy-waiting in main thread

    except sd.PortAudioError as e:
        print(f"\nSoundevice Error: {e}")
        print("Please check your microphone connection and permissions.")
        print("Available devices:")
        print(sd.query_devices())
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
    finally:
        # Signal threads to stop and wait for them
        print("Stopping threads and audio stream...")
        stop_event.set()
        if processor_thread.is_alive():
             processor_thread.join(timeout=5) # Wait max 5 seconds for thread to finish
        print("Application stopped.")