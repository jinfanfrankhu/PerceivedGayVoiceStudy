import os
import subprocess

input_folder = r'C:\Users\jinfa\Desktop\GayStudy\unconvertedWavs'
output_folder = r'C:\Users\jinfa\Desktop\GayStudy\wavs'

# Create output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.lower().endswith('.m4a'):
        base_name = os.path.splitext(filename)[0]
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, base_name + '.wav')

        print(f'Converting: {filename}')
        try:
            subprocess.run([
                'ffmpeg', '-y',
                '-i', input_path,
                '-acodec', 'pcm_s16le',
                '-ar', '44100',
                output_path
            ], check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to convert {filename}: {e}")
        else:
            print(f"✅ Success: {base_name}.wav\n")
            os.remove(input_path)

