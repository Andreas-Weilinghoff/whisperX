##########################################################################################################################
# This python script is designed to implement WhisperX (Bain et al. 2023) on a particular sound file on your machine. 
# The script transforms the diarized output (based on the pyannote diarization) and saves it into a .srt file format.
#
# You can find all the info about WhisperX here: https://github.com/m-bain/whisperX
# You can find the corresponding paper by here: https://www.robots.ox.ac.uk/~vgg/publications/2023/Bain23/bain23.pdf
# 
# This script is distributed under the GNU General Public License.
# Copyright 09/10/2024 by Andreas Weilinghoff.
# You may use/modify this script as you wish. It would just be nice if you cite me:
# Weilinghoff, A. (2024): whisperx_to_srt.py (Version 1.0) [Source code]. https://www.andreas-weilinghoff.com/#code
##########################################################################################################################

import whisperx
import gc 
from datetime import timedelta

# 0. Adapt input parameters
DEVICE = "cpu" 
DIRECTORY = "C:\\Users\\"
FILE_NAME = "testfile"
FILE_EXTENSION = ".wav"
BATCH_SIZE = 16 # reduce if low on GPU mem
COMPUTE_TYPE = "float32" # change to "int8" if low on GPU mem (may reduce accuracy)
LANGUAGE = "en"
WHISPER_MODEL = "medium"

# 1. Get input file and specify name of output file
INPUT_FILE= DIRECTORY + FILE_NAME + FILE_EXTENSION
print(">>>   Input file: " + "'" + str(INPUT_FILE) + "'" + " detected.")
OUTPUT_FILE = str(INPUT_FILE).replace(FILE_EXTENSION,'.srt')

# 2. Transcribe with original whisper (batched)
model = whisperx.load_model(WHISPER_MODEL, DEVICE, compute_type=COMPUTE_TYPE, language=LANGUAGE)
audio = whisperx.load_audio(INPUT_FILE)
result = model.transcribe(audio, batch_size=BATCH_SIZE)
#print(result["segments"]) # before alignment
print(">>>   Whisper transcription complete.")
# delete model if low on GPU resources
# import gc; gc.collect(); torch.cuda.empty_cache(); del model

# 3. Align whisper output
model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=DEVICE)
result = whisperx.align(result["segments"], model_a, metadata, audio, DEVICE, return_char_alignments=False)
#print(result["segments"]) # after alignment
print(">>>   WhisperX alignment complete.")
# delete model if low on GPU resources
# import gc; gc.collect(); torch.cuda.empty_cache(); del model_a

# 4. Assign speaker labels
diarize_model = whisperx.DiarizationPipeline(use_auth_token="YOUR_HUGGING_FACE_ACCESS_TOKEN", device=DEVICE)
# add min/max number of speakers if known
diarize_segments = diarize_model(audio)
# diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)
result = whisperx.assign_word_speakers(diarize_segments, result)
#print(diarize_segments)
#print(result["segments"]) # segments are now assigned speaker IDs
print(">>>   Speaker diarization complete.")

# 5. Convert output into .srt format
result_segments = result["segments"]
#print(result_segments[0])
# Function to convert seconds to SRT time format (HH:MM:SS,ms)
def format_time(seconds):
    td = timedelta(seconds=seconds)
    return str(td)[:-3].replace('.', ',')
# Generate .srt format with speaker labels from diarize_segments
def generate_srt(diarize_segments, result_segments):
    srt_output = []
    segment_index = 0
    srt_index = 1
    # Iterate over result_segments, which contains the utterance text and timings
    for result_segment in result_segments:
        segment_start = result_segment["start"]
        segment_end = result_segment["end"]
        segment_text = result_segment["text"]
        # Find the corresponding speaker from diarize_segments based on the start time of the utterance
        while segment_index < len(diarize_segments) and diarize_segments.iloc[segment_index]["end"] < segment_start:
            segment_index += 1
        if segment_index < len(diarize_segments) and diarize_segments.iloc[segment_index]["start"] <= segment_start <= diarize_segments.iloc[segment_index]["end"]:
            speaker = diarize_segments.iloc[segment_index]["speaker"]
        else:
            speaker = "Unknown"  # Fallback in case no speaker is found
        # Format the SRT block for the entire utterance
        start_time = format_time(segment_start)
        end_time = format_time(segment_end)
        srt_output.append(f"{srt_index}")
        srt_output.append(f"{start_time} --> {end_time}")
        srt_output.append(f"{speaker}: {segment_text.strip()}")
        srt_output.append("")  # Empty line to separate subtitles
        srt_index += 1  # Increment SRT block index
    return "\n".join(srt_output)
# Generate SRT content
srt_content = generate_srt(diarize_segments, result_segments)
# Write to an .srt file
with open(OUTPUT_FILE, 'w') as f:
    f.write(srt_content)
print(">>>   SRT file generated successfully.")