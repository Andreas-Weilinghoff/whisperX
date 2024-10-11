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
# Weilinghoff, A. (2024): whisperx_to_textgrid.py (Version 1.0) [Source code]. https://www.andreas-weilinghoff.com/#code
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
OUTPUT_FILE = str(INPUT_FILE).replace(FILE_EXTENSION,'.TextGrid')

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

# 5. Convert output into .TextGrid format
result_segments = result["segments"]
#print(result_segments[0])

def textgrid_header(endtime, tiers_total):
    return f'File type = "ooTextFile"\n' \
           f'Object class = "TextGrid"\n\n' \
           f'xmin = 0\n' \
           f'xmax = {endtime}\n' \
           f'tiers? <exists>\n' \
           f'size = {tiers_total}\n' \
           f'item []:\n'

def textgrid_tier_header(tier_index, speaker_name, endtime, intervals_count):
    return f'\titem [{tier_index}]:\n' \
           f'\t\tclass = "IntervalTier"\n' \
           f'\t\tname = "{speaker_name}"\n' \
           f'\t\txmin = 0\n' \
           f'\t\txmax = {endtime}\n' \
           f'\t\tintervals: size = {intervals_count}'

def textgrid_item(xmin, xmax, text, interval_number):
    return f'\t\tintervals [{interval_number}]:\n' \
           f'\t\t\txmin = {xmin}\n' \
           f'\t\t\txmax = {xmax}\n' \
           f'\t\t\ttext = \"{text.strip()}\"'

def build_textgrid(diarize_segments, result_segments, endtime):
    # Initialize the content
    speakers = diarize_segments['speaker'].unique()  # Get unique speakers
    tiers_total = len(speakers)  # One tier per speaker
    
    content = [textgrid_header(endtime, tiers_total)]
    pause_threshold = 0.1  # Define a minimum pause threshold (in seconds)
    
    # Create a tier for each speaker
    for idx, speaker in enumerate(speakers, 1):
        speaker_segments = diarize_segments[diarize_segments['speaker'] == speaker]
        intervals = []
        previous_end = 0.0  # Start from time 0
        
        for i, segment in speaker_segments.iterrows():
            xmin = segment['start']
            xmax = segment['end']
            
            # Insert an empty interval only if the gap is larger than the threshold
            if xmin - previous_end > pause_threshold:
                intervals.append(textgrid_item(previous_end, xmin, "", len(intervals) + 1))
            
            # Find the corresponding text for the time range in result_segments
            text = ""
            for result_segment in result_segments:
                # Check if there is overlap between the transcription and the speaker segment
                if not (result_segment['end'] < xmin or result_segment['start'] > xmax):
                    # Add the text if it overlaps with the speaker's segment
                    text += result_segment['text'] + " "
            
            # Add the interval with text
            intervals.append(textgrid_item(xmin, xmax, text.strip(), len(intervals) + 1))
            
            # Update previous_end to the current xmax
            previous_end = xmax
        
        # Add an empty interval if the last xmax is less than the total endtime and the gap is significant
        if endtime - previous_end > pause_threshold:
            intervals.append(textgrid_item(previous_end, endtime, "", len(intervals) + 1))
        
        # Add the header for the speaker's tier
        content.append(textgrid_tier_header(idx, speaker, endtime, len(intervals)))
        content.extend(intervals)  # Add all intervals to the content
    
    # Join the content and return
    return '\n'.join(content)



# Example usage:
endtime = max(result_segment["end"] for result_segment in result["segments"])  # Calculate endtime from the last result segment
textgrid_content = build_textgrid(diarize_segments, result["segments"], endtime)

# Save the TextGrid file
with open(OUTPUT_FILE, 'w') as f:
    f.write(textgrid_content)
print(">>>   TextGrid file generated successfully.")