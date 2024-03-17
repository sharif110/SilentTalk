import os
import json

# Get the absolute path of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the absolute path to the 'WLASL/videos' directory
videos_dir = os.path.join(script_dir, 'C:\Users\Pc\Desktop\WLASL\videos')

# Check if the directory exists
if os.path.exists(videos_dir):
    filenames = set(os.listdir(videos_dir))
else:
    print("The 'WLASL/videos' directory was not found at:", videos_dir)
    filenames = set()  # Create an empty set

content = json.load(open('WLASL_v0.3.json'))

missing_ids = []

for entry in content:
    instances = entry['instances']

    for inst in instances:
        video_id = inst['video_id']
        if video_id + '.mp4' not in filenames:
            missing_ids.append(video_id)

with open('missing.txt', 'w') as f:
    f.write('\n'.join(missing_ids))
