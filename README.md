# Head Changer

# Based on
https://github.com/yinguobing/head-pose-estimation

# How to use

## Sample Data

Please refer to `/sample-data`

## Create head changed serial number frame images

```
sudo docker build -t redshoga/head-changer .
sudo docker run -it -v $(pwd):/share redshoga/head-changer python /share/changer.py --video /share/sample-data/video1.mp4 --image /share/sample-data/4.png --output-dir /share/output
```

## Transfer frame images to video

```
ffmpeg -r 30 -i ./output/%d.png -vcodec libx264 -pix_fmt yuv420p -r 30 ./sample-data/out.mp4
```

# TODO
- [ ] Add example gif video to README.md
- [ ] Refactor scripts
- [ ] Add transfer frame images to video using FFmpeg in script
