from projects.qwenvl_sam2.datasets.Charades_Dataset import CharadesDataset
import json
from collections import defaultdict
import os

class ActivityNetDataset(CharadesDataset):

    def json_file_preprocess(self, expression_file):
        # prepare expression annotation files
        with open(expression_file, 'r') as f:
            expression_datas = json.load(f)

        vid2metaid = defaultdict(list)
        for sample_info in expression_datas:
            video_name = sample_info["video"]
            timestamps = sample_info["timestamps"]
            duration = sample_info["duration"]
            sentences = sample_info["sentences"]
            assert len(timestamps) == len(sentences)
            video_name = self.get_mp4_filename(video_name)
            for i in range(len(sentences)):
                sample_dict = {}
                sample_dict["timestamp"] = timestamps[i]
                sample_dict["caption"] = sentences[i]
                sample_dict["duration"] = duration
                vid2metaid[video_name].append(sample_dict)
        return vid2metaid

    def get_mp4_filename(self, filepath):
        base, ext = os.path.splitext(filepath)
        if ext.lower() != '.mp4':
            return base + '.mp4'
        return filepath

