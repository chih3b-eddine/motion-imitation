# Clone Vibe and install its requirements
!git clone https://github.com/mkocabas/VIBE.git
%cd VIBE/
!pip install torch==1.4.0 numpy==1.17.5
!pip install -r requirements.txt
!source scripts/prepare_data.sh


# Download the youtube video with the given ID
YOUTUBE_ID = 'wDCVnVpcWAs'  # 01:46
!youtube-dl  -f 'bestvideo[ext=mp4]' --output "youtube.%(ext)s" https://www.youtube.com/watch?v=$YOUTUBE_ID

# cut 6 seconds starting from moment
!ffmpeg -y -loglevel info -i youtube.mp4 -ss 00:01:46 -t 6 video.mp4
        
# run Vibe
!python demo.py --vid_file ../video.mp4 --output_folder ../data/  