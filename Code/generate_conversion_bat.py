import os

input_folder = r'C:\Users\jinfa\Desktop\GayStudy\m4as'
output_folder = r'C:\Users\jinfa\Desktop\GayStudy\wavs'
bat_file_path = 'convert_all.bat'

with open(bat_file_path, 'w', encoding='utf-8') as bat_file:
    bat_file.write('@echo off\n')
    bat_file.write('echo Starting conversion...\n\n')

    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.m4a'):
            base_name = os.path.splitext(filename)[0]
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, base_name + '.wav')

            # Wrap in quotes to protect against spaces
            command = f'ffmpeg -y -i "{input_path}" -acodec pcm_s16le -ar 44100 "{output_path}"\n'
            bat_file.write(command)

    bat_file.write('\necho Conversion complete.\npause\n')

print(f"Batch file written to: {bat_file_path}")
