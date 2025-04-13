# realtimeASR-whisper
Realtime ASR from any input with WhisperFile and python


### Introduction
I have tried for months to have a realtime translation app

I know there are some existing one, probably better than this project.

The POC was to get it done with coding assistnce, in my case with Gemin1 2.5 Pro.

#### initial prompt
```
i want to create a python application for real time automatic speech recognition. consider the audio coming from the microphone, I want to be able to handle long recordings, so there must be a continuous 15 seconds audio chunks with 2 seconds overlap to be sent to a whisper.cpp srver API.
```

#### Requirements
```
pip install sounddevice numpy requests scipy openai gradio tiktoken
```
The last three are for future GUI implementation
- You have a whisper.cpp server running and accessible via HTTP. We'll use a placeholder URL (http://localhost:8080/inference) which you should replace with your actual server endpoint. The standard whisper.cpp server example often uses /inference or /asr. Check your server's documentation.
- The whisper.cpp server expects audio data via a POST request, likely as a file upload (e.g., multipart/form-data) in WAV format.
- Whisper models generally work best with 16kHz mono audio. We will configure the microphone input accordingly.

#### Core Logic:

1. Audio Capture: Use sounddevice.InputStream to continuously capture audio from the default microphone at 16kHz mono.
2. Buffering: Store the incoming audio samples in a buffer (a NumPy array).
2. Chunking & Overlapping:
    - Define chunk duration (15 seconds) and overlap duration (2 seconds).
    - Calculate corresponding sample counts.
    - Run a separate thread that continuously monitors the buffer.
    - When the buffer holds at least 15 seconds of audio, extract the latest 15-second chunk.
    - Send this chunk to the API.
    - After processing, discard the oldest 13 seconds (15s chunk - 2s overlap) from the buffer, keeping the last 2 seconds for the next chunk's overlap.
4. API Interaction:
    - Format the extracted audio chunk as a WAV file in memory.
    - Use the requests library to send this WAV data as a file in a POST request to the whisper.cpp server URL.
    - Receive and print the transcription result.
5 Threading: Use Python's threading module to run the audio processing (chunking, API calls) in the background so it doesn't block audio capture. Use a threading.Lock to safely access the shared audio buffer from both the audio callback and the processing thread.


```
thanks it works. Is there a way to route any windows audio to the microphone input? In this way it is not required to have a speaker in person, and I can transcribe any media running.
```

Okay, yes, routing your computer's audio output directly into an audio input (like a virtual microphone) is a common requirement for tasks like this. It avoids the need for a physical microphone and speakers and captures the audio digitally without background noise.

There are primarily two ways to achieve this on Windows:

Method 1: Using Built-in "Stereo Mix" (or similar)

Method 2: Using Virtual Audio Cable Software

If "Stereo Mix" isn't available, the most reliable alternative is to use virtual audio cable software. This software creates virtual audio devices (inputs and outputs) that you can use to route audio between applications.   

Recommended Software: VB-Audio Virtual Cable (VB-CABLE)

- Website: https://vb-audio.com/Cable/
- Cost: Free (Donationware)
- How it works: It installs one virtual input device ("CABLE Input") and one virtual output device ("CABLE Output").

Steps:
1. Install VB-CABLE: Download and install it from the official website (run the setup as administrator). Reboot your computer if prompted.
<img src='https://github.com/fabiomatricardi/realtimeASR-whisper/raw/main/VA-config001.png' width=500>
2. Route System Audio Output:
    - Right-click the speaker icon in the system tray.
    - Select "Sounds" or "Open Sound settings".
    - Go to the "Playback" tab (or select the main Output device in Windows 10/11 Settings).
    - Find "CABLE Input (VB-Audio Virtual Cable)".
    - Right-click on it and select "Set as Default Device".
    - Result: Now, all standard Windows audio output will be sent to this virtual device instead of your actual speakers/headphones. You won't hear the audio through your speakers anymore unless you set up monitoring (see below).

4. Modify Your Python Script to Use the Virtual Input:
    - The virtual output of the cable acts as the virtual input microphone. This device is usually named "CABLE Output (VB-Audio Virtual Cable)".
    - Your Python script needs to be explicitly told to use this specific device instead of the default.


#### How to Hear Audio While Using VB-CABLE (Optional Monitoring):

Since the system audio is now routed to the virtual cable, you won't hear it. To hear it:
<img src='https://github.com/fabiomatricardi/realtimeASR-whisper/raw/main/VA-config002.png' width=500>
1. Go back to the Sound Control Panel -> Recording tab.
2. Find "CABLE Output (VB-Audio Virtual Cable)", right-click it, and choose "Properties".
3. Go to the "Listen" tab.
4. Check the box "Listen to this device".
5. From the "Playback through this device" dropdown, select your actual speakers or headphones (e.g., "Speakers (Realtek High Definition Audio)").
6. Click "Apply" or "OK".

You should now hear the system audio again, possibly with a very slight delay, while it's also being fed into the virtual cable for your script.




### [WhisperFILE](https://github.com/Mozilla-Ocho/llamafile/blob/main/whisper.cpp/doc/server.md)
A project from llamafile - Mozilla Foundation

Whisperfile is a high-performance implementation of OpenAI's Whisper model that's based off the whisper.cpp software written by Georgi Gerganov, et al.

Full info [here](https://github.com/Mozilla-Ocho/llamafile/blob/main/whisper.cpp/doc/index.md)

##### [Translation](https://github.com/Mozilla-Ocho/llamafile/blob/main/whisper.cpp/doc/translate.md)
Whisperfile is not only able to transcribe speech to text. It's able to translate that speech into English too, at the same time. All you have to do is pass the `-tr` or `--translate` flag.

Choosing a Model
In order for translation to work, you need to be using a multilingual model. On https://huggingface.co/ggerganov/whisper.cpp/ the files that have `.en` in the name are English-only; you can't use those for translation. One model that does work well in translation mode is `ggml-medium-q5_0.bin`.

Language Override
By default, the source language will be auto-detected. This works great except for recordings with multiple languages. For example, if you have a recording with a little bit of English at the beginning, but the rest is in French, then you may want to pass the -l fr flag, to explicitly specify the source language as French.



In my case I want to ASR and translate from Mandarin Chinese to English
```
whisperfile-0.9.2.exe --server -l zh -tr -m .\ggml-base-q5_1.bin
```

##### Options
you can find more info from [source - Llamafile](https://github.com/Mozilla-Ocho/llamafile/blob/main/whisper.cpp/doc/server.md)
```
options:
  -h,        --help              [default] show this help message and exit
  -t N,      --threads N         [4      ] number of threads to use during computation
  -p N,      --processors N      [1      ] number of processors to use during computation
  -ot N,     --offset-t N        [0      ] time offset in milliseconds
  -on N,     --offset-n N        [0      ] segment index offset
  -d  N,     --duration N        [0      ] duration of audio to process in milliseconds
  -mc N,     --max-context N     [-1     ] maximum number of text context tokens to store
  -ml N,     --max-len N         [0      ] maximum segment length in characters
  -sow,      --split-on-word     [false  ] split on word rather than on token
  -bo N,     --best-of N         [2      ] number of best candidates to keep
  -bs N,     --beam-size N       [-1     ] beam size for beam search
  -wt N,     --word-thold N      [0.01   ] word timestamp probability threshold
  -et N,     --entropy-thold N   [2.40   ] entropy threshold for decoder fail
  -lpt N,    --logprob-thold N   [-1.00  ] log probability threshold for decoder fail
  -debug,    --debug-mode        [false  ] enable debug mode (eg. dump log_mel)
  -tr,       --translate         [false  ] translate from source language to english
  -di,       --diarize           [false  ] stereo audio diarization
  -tdrz,     --tinydiarize       [false  ] enable tinydiarize (requires a tdrz model)
  -nf,       --no-fallback       [false  ] do not use temperature fallback while decoding
  -ps,       --print-special     [false  ] print special tokens
  -pc,       --print-colors      [false  ] print colors
  -pr,       --print-realtime    [false  ] print output in realtime
  -pp,       --print-progress    [false  ] print progress
  -nt,       --no-timestamps     [false  ] do not print timestamps
  -l LANG,   --language LANG     [en     ] spoken language ('auto' for auto-detect)
  -dl,       --detect-language   [false  ] exit after automatically detecting language
             --prompt PROMPT     [       ] initial prompt
  -m FNAME,  --model FNAME       [models/ggml-base.en.bin] model path
  -oved D,   --ov-e-device DNAME [CPU    ] the OpenVINO device used for encode inference
  --host HOST,                   [127.0.0.1] Hostname/ip-adress for the server
  --port PORT,                   [8080   ] Port number for the server
```


### Quantized Whisper models
Download from the [official GGerganov repo](https://huggingface.co/ggerganov/whisper.cpp/tree/main)

<img src='https://github.com/fabiomatricardi/realtimeASR-whisper/raw/main/audiomodels.png' width=400>

Models are multilingual unless the model name includes .en. Models ending in -q5_0 are quantized. Models ending in -tdrz support local diarization (marking of speaker turns) using tinydiarize. More information about models is available upstream (openai/whisper). The list above is a subset of the models supported by the download-ggml-model.sh script, but many more are available at https://huggingface.co/ggerganov/whisper.cpp/tree/main and elsewhere.



### Inspirations
- YOUTUBE VIDEO   [https://www.youtube.com/watch?v=_spinzpEeFM](https://www.youtube.com/watch?v=_spinzpEeFM)

- GITHUB whisper_streaming PROJECT
[https://github.com/ufal/whisper_streaming](https://github.com/ufal/whisper_streaming)





### Additional Software reqruied

<img src='https://github.com/fabiomatricardi/realtimeASR-whisper/raw/main/VA-VBAudio.png' width=400> <img src='https://github.com/fabiomatricardi/realtimeASR-whisper/raw/main/VA-VirtualAudioCable.png' width=400>



