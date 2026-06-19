import json
from collections import defaultdict
import os

image_dir = "data/VTG_data/videos_1FPS"


def get_mp4_filename(filepath):
    base, ext = os.path.splitext(filepath)
    if ext.lower() != '.mp4':
        return base + '.mp4'
    return filepath


def json_file_preprocess(expression_file):
    # prepare expression annotation files
    with open(expression_file, 'r') as f:
        expression_datas = json.load(f)

    vid2metaid = defaultdict(list)
    for sample_info in expression_datas:
        sample_dict = {}
        video_name = sample_info["video"]
        timestamps = sample_info["timestamps"]
        duration = sample_info["duration"]
        sentences = sample_info["sentences"]
        video_name = get_mp4_filename(video_name)
        if not os.path.exists(os.path.join(image_dir, video_name)):
            print(video_name)
        assert len(timestamps) == len(sentences)
        for i in range(len(sentences)):
            sample_dict["timestamp"] = timestamps[i]
            sample_dict["caption"] = sentences[i]
            sample_dict["duration"] = duration
            vid2metaid[video_name].append(sample_dict)
    return vid2metaid


if __name__ == "__main__":
    expression_file = "data/VTG_data/activitynet_captions_train.json"
    vid2metaid = json_file_preprocess(expression_file)
