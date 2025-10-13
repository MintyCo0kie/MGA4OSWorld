import pysrt
import subprocess

def compare_srt_files(file1, file2):
    header_path = "/home/user/"
    subs1 = pysrt.open(header_path+file1)
    subs2 = pysrt.open(header_path+file2)

    return True if subs1 == subs2 else False 

def check_mp4file_subtitles_is_moved(file):
    # execute "ffmpeg -i ${file}" command, check if there is no subtitle stream in the output
    # if there is no subtitle stream, return True, else return False    
    result = subprocess.run(['ffmpeg', '-i', file], capture_output=True, text=True)
    output = result.stderr

    # Check if there are subtitle streams
    if "Stream #0:2[0x3](und): Subtitle" in output:
        return False
    else:
        return True
    
if (compare_srt_files('subtitles.srt', 'subtitles_Gold.srt') and check_mp4file_subtitles_is_moved('video.mp4')):
    print("true")
else:
    print("false")
