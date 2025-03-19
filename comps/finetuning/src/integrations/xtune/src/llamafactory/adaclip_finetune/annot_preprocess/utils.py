import os


def is_exist(video_root, video_id):
    extensions = ["mp4", "mkv", "webm", "avi"]
    for ext in extensions:
        if os.path.isfile(os.path.join(video_root, video_id + "." + ext)):
            return ext
    return ''