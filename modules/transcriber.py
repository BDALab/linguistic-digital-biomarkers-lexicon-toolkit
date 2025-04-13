import whisper
import torch
import os

from docx import Document
from tqdm import tqdm
import time

# In[]

recordings_folder_path = "recordings_study"
audio_format = ".wav"

model_name = "large-v2"
device = "cuda"  # "cpu"

# In[]

model = whisper.load_model(model_name).to(device)
options = whisper.DecodingOptions(fp16=False)

# In[]

n_recs = len(os.listdir(recordings_folder_path))

start_time = time.time()

for filename in tqdm(os.listdir(recordings_folder_path)):

    if filename.endswith(audio_format):
        id = os.path.splitext(filename)[0]

        result = model.transcribe(audio=recordings_folder_path + "/" + filename, language="Czech")

        # In[]

        document = Document()
        document.add_paragraph(result["text"])
        document.save(recordings_folder_path + "/" + id + '.docx')

# In[]

end_time_transcription = time.time()
time_transcription = end_time_transcription - start_time
print("Transcription finished. Time: ", time_transcription / 60, " min")
