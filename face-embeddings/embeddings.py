from deepface import DeepFace
import glob
import time
import json
import random
import os

amount = int(os.environ.get("AMOUNT") or 20)
model = os.environ.get("MODEL") or "VGG-Face"
library = os.environ.get("LIBRARY") or "/library"

glob_path = f"{library}/**/*.jpg"

DeepFace.build_model(model)

allFiles = [path for path in glob.glob(glob_path, recursive=True) if ("thumb" not in path)]

print(f"Found {len(allFiles)} jpg files")

files = random.sample(allFiles, amount)

print(f"Sampled {amount} files for processing")

embeddings = []
faceTimes = []
skipTimes = []

for file in files:
  start = time.time()
  try:
    e = DeepFace.represent(img_path = file)
    end = time.time()
    embeddings += e
    faceTimes.append(end - start)
  except:
    end = time.time()
    skipTimes.append(end - start)

faceAvg = sum(faceTimes) / len(faceTimes)
skipAvg = sum(skipTimes) / len(skipTimes)

print(f"{len(faceTimes)} files had a face. {len(skipTimes)} files did not and were skipped")
print(f"Average time per file with faces: {faceAvg} seconds")
print(f"Average time per file without faces: {skipAvg} seconds")

serialized = json.dumps(embeddings)
avg_bytes = len(serialized.encode("utf-8")) / len(embeddings)

print(f"Average size per embedding (as json): {avg_bytes} bytes")