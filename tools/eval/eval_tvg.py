import json
import os
import time
import sys
import argparse
import pdb

def read_json(path):
    with open(path, "r") as fin:
        datas = json.load(fin)
    return datas


def iou(A, B):
    max0 = max((A[0]), (B[0]))
    min0 = min((A[0]), (B[0]))
    max1 = max((A[1]), (B[1]))
    min1 = min((A[1]), (B[1]))
    return max(min1 - max0, 0) / (max1 - min0)


def toSec(timeStr):
    t = time.strptime(timeStr, "%H:%M:%S")
    return t.tm_hour * 3600 + t.tm_min * 60 + t.tm_sec

def captiondata_modify(steps):
    modify_data = {}
    for i, step in enumerate(steps[0]):
        for key in step["step"].keys():
            name = step["step"][key]["query_idx"]
            modify_data[name] = [[step['step'][key]["startime"], step['step'][key]["endtime"]]]
        
    return modify_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pred_file", type=str)
    parser.add_argument('--gt_file', type=str, default='data/VTG_data/test.caption_coco_format.json')
    parser.add_argument('--sample', action='store_true', default=False)
    parser.add_argument("--save_name", type=str, default="charades.json")
    args = parser.parse_args()
    '''
    {
        "query_idx": [start_time, end_time],
        ...
    }
    '''
    # answer = read_json(args.gt_file)
    # answer = answer["annotations"]
    # gt_timestamps = {}
    # for jterm in answer:
    #     gt_timestamps[jterm["id"]] = jterm["timestamp"]
        
    # submission = read_json(args.pred_file)
    # pred_timestamps = {}
    # for qid, jterm in submission.items():
    #     pred_timestamps[int(qid)] = jterm["timestamp"]


    submission = read_json(args.pred_file)
    gt_timestamps = {}
    pred_timestamps = {}
    for item in submission:
        gt_timestamps[item["id"]] = [item["gt_start"], item["gt_end"]]
        pred_timestamps[item["id"]] = [float(item["pred_start"]), float(item["pred_end"])]
    
    if args.sample:
        new = {}
        for qid in pred_timestamps.keys():
            new[qid] = gt_timestamps[qid]
        gt_timestamps = new
    num = len(gt_timestamps)
    print(f"# pred video timestamps {len(pred_timestamps)}; # gt video timestamps {len(gt_timestamps)}")
    assert len(gt_timestamps) == len(pred_timestamps)
    Result = {0.3:0, 0.5:0, 0.7:0}
    total_iou = []
    for c_iou in [0.3, 0.5, 0.7]:
        for key in gt_timestamps.keys():
            if len(pred_timestamps[key]) < 1:
                continue
            this_iou = iou(gt_timestamps[key], pred_timestamps[key])
            if(this_iou) >= c_iou:
                Result[c_iou] = Result[c_iou] + 1
            total_iou.append(this_iou)
    mIOU = sum(total_iou)/(len(total_iou)+1e-6)
    results = "IOU 0.3: {0}\nIOU 0.5: {1}\nIOU 0.7: {2}\nmIOU: {3}%".format(Result[0.3]*100/num, Result[0.5]*100/num, Result[0.7]*100/num, mIOU*100)
    print(results)
    output_path = os.path.join(os.path.dirname(args.pred_file), args.save_name)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)