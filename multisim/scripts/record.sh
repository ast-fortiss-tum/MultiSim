
ffmpeg -video_size 1680x1050 -framerate 25 -f x11grab -i :1.0 multi_sim_demo.mp4
ffmpeg -i multi_sim_demo.mp4 -vf scale=1280:720 -q:v 1 -q:a 1 multi_sim_demo.wmv
