"""
Planning evaluation script with optional classification accuracy evaluation.

This script evaluates:
1. Planning metrics (L2 error, collision rates, boundary violations)
2. Optional: Classification accuracy (requires --eval_classification flag)

The classification accuracy evaluation looks for *_classify_acc.txt files 
containing 'True' or 'False' values indicating whether the classification
prediction was correct for each sample.
"""

import argparse
import pickle
import os
import numpy as np
from nuscenes.eval.common.utils import Quaternion
import json
from os import path as osp
from planning_utils import PlanningMetric
import torch
from tqdm import tqdm
import threading
import re
import copy

id_dist = {0: -10.753771146138508, 1: -9.59672942528358, 2: -8.59819254381903, 3: -7.904938940274514, 
4: -7.285219575229442, 5: -6.701759373148276, 6: -6.228400120368372, 7: -5.790760440826412, 
8: -5.425686396561655, 9: -5.0828556467275146, 10: -4.744958385006411, 11: -4.3752186652457326, 12: -4.0128931058656185, 
13: -3.7105577418776345, 14: -3.461096053798445, 15: -3.1997231754602184, 16: -2.9541410628487084, 17: -2.6827685074683227, 
18: -2.405682117187954, 19: -2.1267487608383764, 20: -1.8744070736647487, 21: -1.6513011703888587, 22: -1.4579966144952161, 
23: -1.2848656746057365, 24: -1.1325989551795468, 25: -0.9923438125115664, 26: -0.8660390488028753, 27: -0.7475322028861209, 
28: -0.6360212394114244, 29: -0.5371739974955956, 30: -0.4469941846194718, 31: -0.3650487743479225, 32: -0.2918444662400148, 
33: -0.22680273552282149, 34: -0.1696966372738311, 35: -0.1187845557495768, 36: -0.07241398952937672, 37: -0.03310337285550613, 
38: -0.0015949679265290229, 39: 0.03176164131575465, 40: 0.07314882512060539, 41: 0.12302191168061238, 42: 0.18015049105854342, 
43: 0.24977361015417987, 44: 0.33070564727657814, 45: 0.4224969216287118, 46: 0.523392403215317, 47: 0.6256079429637378, 
48: 0.7287287656342247, 49: 0.8295795130882144, 50: 0.932965743170842, 51: 1.0423079060008176, 52: 1.1542496034559209, 
53: 1.2649599019982407, 54: 1.3854073257510229, 55: 1.5146264115757968, 56: 1.6534584416037164, 57: 1.7910500739189152, 
58: 1.9336565667362287, 59: 2.072826230705303, 60: 2.2115457209394718, 61: 2.3398350067751856, 62: 2.4661438606774784, 
63: 2.5853939706628966, 64: 2.705099730101069, 65: 2.830465933926793, 66: 2.9550756021639772, 67: 3.074016864904779, 
68: 3.186821440087278, 69: 3.2970794350554873, 70: 3.4069461729649557, 71: 3.5207698326875736, 72: 3.6379030045219087, 
73: 3.7623763780915334, 74: 3.884859606555608, 75: 4.011348465314279, 76: 4.136568051312877, 77: 4.256756281739728, 
78: 4.375849092893766, 79: 4.488329148996732, 80: 4.598593553942785, 81: 4.705343488437026, 82: 4.820401088844323, 
83: 4.937422963439441, 84: 5.058890364386818, 85: 5.187730544150059, 86: 5.310740662296659, 87: 5.430726752383929, 
88: 5.545603264492565, 89: 5.6517278370411805, 90: 5.754543849916169, 91: 5.863232715281448, 92: 5.972506479527746, 
93: 6.095268937503045, 94: 6.227488088251559, 95: 6.37948853165299, 96: 6.542146583449258, 97: 6.7166037978989, 
98: 6.892884731715451, 99: 7.080059467295033, 100: 7.257359217624273, 101: 7.430606667523902, 102: 7.594698996530088, 
103: 7.755356911195807, 104: 7.909185194690471, 105: 8.059485538740022, 106: 8.209712461233138, 107: 8.35514352085826, 
108: 8.499879161333705, 109: 8.647526895208582, 110: 8.811711206155662, 111: 8.97707302644929, 112: 9.13482952432318, 
113: 9.275043151566006, 114: 9.407433576238521, 115: 9.528953758556725, 116: 9.648095205659775, 117: 9.763136813359228, 
118: 9.88578972671971, 119: 10.023127060322334, 120: 10.16865114882441, 121: 10.319087035447648, 122: 10.475032242861658, 
123: 10.633334187825518, 124: 10.788813108264812, 125: 10.962027334930875, 126: 11.129296838436662, 127: 11.301415445786844, 
128: 11.474560759961603, 129: 11.635874632025967, 130: 11.7932517600782, 131: 11.964559966397571, 132: 12.138875299134586, 
133: 12.312952508057187, 134: 12.484664617471362, 135: 12.654259835098154, 136: 12.817338553972027, 137: 12.97223679270456, 
138: 13.132402437357245, 139: 13.281995034217838, 140: 13.433746889450244, 141: 13.576767158180013, 142: 13.72127544120749, 
143: 13.86934523630624, 144: 14.025564163450209, 145: 14.180257716689537, 146: 14.344183065260653, 147: 14.521548401819514, 
148: 14.693983916368124, 149: 14.877313491660106, 150: 15.058887256616545, 151: 15.218794870601158, 152: 15.36906867858778, 
153: 15.518457196508852, 154: 15.669258948295345, 155: 15.823414134808036, 156: 15.988689613342283, 157: 16.16233137350167, 
158: 16.33446017149867, 159: 16.528238395431245, 160: 16.72460580804912, 161: 16.91952348249761, 162: 17.1284894716172, 
163: 17.339478359130275, 164: 17.53737921371037, 165: 17.719437660138635, 166: 17.899441763858672, 167: 18.091391273938964, 
168: 18.279078708402107, 169: 18.449508178979162, 170: 18.636386441562877, 171: 18.831668537003658, 172: 19.04166718114174, 
173: 19.269774564324997, 174: 19.491631466409437, 175: 19.690695552905076, 176: 19.882089722976993, 177: 20.079936568573878, 
178: 20.26065455672211, 179: 20.438695765932934, 180: 20.630103880938368, 181: 20.820761150783966, 182: 21.018763873788668, 
183: 21.215271052493847, 184: 21.40290150497899, 185: 21.58127604715923, 186: 21.76909218343099, 187: 21.97825696507808, 
188: 22.211315707383655, 189: 22.46632709744611, 190: 22.70706443786622, 191: 22.938350272113684, 192: 23.16415346385353, 
193: 23.376872426703912, 194: 23.58188770626401, 195: 23.793671938837804, 196: 23.990983875318502, 197: 24.180191117459124, 
198: 24.397498102471385, 199: 24.640973915656414, 200: 24.908912198296907, 201: 25.17465356805108, 202: 25.457113962588103, 
203: 25.743075688680015, 204: 26.01100498338871, 205: 26.313981787476024, 206: 26.614896953969765, 207: 26.923727797011463, 
208: 27.226505361178507, 209: 27.47671459411002, 210: 27.724875926971436, 211: 27.96278130877149, 212: 28.208863905746576, 
213: 28.462221581258895, 214: 28.73334528241839, 215: 29.018578716056066, 216: 29.313470626018457, 217: 29.607951220344106, 
218: 29.877527542603325, 219: 30.192061344782516, 220: 30.54418698727098, 221: 30.907276479209337, 222: 31.268708368627042, 
223: 31.67044461059571, 224: 32.04273581273347, 225: 32.39796045775056, 226: 32.80073779097227, 227: 33.21362167358398, 
228: 33.60689970504406, 229: 34.08469395465161, 230: 34.55107620616019, 231: 35.11154948801234, 232: 35.77493242536272, 
233: 36.30785266423629, 234: 36.88883959592043, 235: 37.52514742337739, 236: 38.207706451416, 237: 38.9694640295846, 
238: 39.75810018686147, 239: 40.47837837537128, 240: 41.37829780578612, 241: 42.250344238281244, 242: 43.07227046673114, 
243: 43.65122580528259, 244: 44.21925196928136, 245: 44.97597408294676, 246: 45.786103610334706, 247: 46.71294498443602, 
248: 47.434690128673196, 249: 48.171729087829576, 250: 49.05333862304688, 251: 50.1207661098904, 252: 50.94644737243652, 
253: 51.80799560546875, 254: 53.047519683837876, 255: 54.65858046213785}

def append_tangent_directions(traj):
    directions = []
    directions.append(np.arctan2(traj[0][1], traj[0][0]))
    for i in range(1, len(traj)):
        vector = traj[i] - traj[i-1]
        angle = np.arctan2(vector[1], vector[0])
        directions.append(angle)
    directions = np.array(directions).reshape(-1, 1)
    traj_yaw = np.concatenate([traj, directions], axis=-1)
    return traj_yaw

def print_progress(current, total):
    percentage = (current / total) * 100
    print(f"\rProgress: {current}/{total} ({percentage:.2f}%)", end="")

def process_data(preds, start, end, key_infos, metric_dict, lock, pbar, planning_metric):
    ego_boxes = np.array([[0.5 + 0.985793, 0.0, 0.0, 4.08, 1.85, 0.0, 0.0, 0.0, 0.0]])
    for i in range(start, end):
        try:
            data = key_infos['infos'][i]
            if data['token'] not in preds.keys():
                continue
            pred_traj = preds[data['token']]
            gt_traj, mask = data['gt_planning'], data['gt_planning_mask'][0]

            gt_agent_boxes = np.concatenate([data['gt_boxes'], data['gt_velocity']], -1)
            gt_agent_feats = np.concatenate([data['gt_fut_traj'][:, :6].reshape(-1, 12), data['gt_fut_traj_mask'][:, :6], data['gt_fut_yaw'][:, :6], data['gt_fut_idx']], -1)
            bev_seg = planning_metric.get_birds_eye_view_label(gt_agent_boxes, gt_agent_feats, add_rec=True)

            e2g_r_mat = Quaternion(data['ego2global_rotation']).rotation_matrix
            e2g_t = data['ego2global_translation']
            drivable_seg = planning_metric.get_drivable_area(e2g_t, e2g_r_mat, data)

            pred_traj_yaw = append_tangent_directions(pred_traj[..., :2])
            pred_traj_mask = np.concatenate([pred_traj_yaw[..., :2].reshape(1, -1), np.ones_like(pred_traj_yaw[..., :1]).reshape(1, -1), pred_traj_yaw[..., 2:].reshape(1, -1)], axis=-1)
            ego_seg = planning_metric.get_ego_seg(ego_boxes, pred_traj_mask, add_rec=True)

            pred_traj = torch.from_numpy(pred_traj).unsqueeze(0)
            gt_traj = torch.from_numpy(gt_traj[..., :2])
            fut_valid_flag = mask.all()
            future_second = 3
            if fut_valid_flag:
                with lock:
                    metric_dict['samples'] += 1
                for i in range(future_second):
                    cur_time = (i+1)*2
                    ade = float(
                        sum(
                            np.sqrt(
                                (pred_traj[0, i, 0] - gt_traj[0, i, 0]) ** 2
                                + (pred_traj[0, i, 1] - gt_traj[0, i, 1]) ** 2
                            )
                            for i in range(cur_time)
                        )
                        / cur_time
                    )
                    metric_dict['l2_{}s'.format(i+1)] += ade
                    
                    obj_coll, obj_box_coll = planning_metric.evaluate_coll(pred_traj[:, :cur_time], gt_traj[:, :cur_time], torch.from_numpy(bev_seg[1:]).unsqueeze(0))
                    metric_dict['plan_obj_box_col_{}s'.format(i+1)] += obj_box_coll.max().item()
                    
                    rec_out = ((np.expand_dims(drivable_seg, 0) == 0) & (ego_seg[0:1] == 1)).sum() > 0
                    out_of_drivable = ((np.expand_dims(drivable_seg, 0) == 0) & (ego_seg[1:cur_time+1] == 1)).sum() > 0
                    if out_of_drivable and not rec_out:
                        metric_dict['plan_boundary_{}s'.format(i+1)] += 1
            pbar.update(1)
        except Exception as e:
            print(e)

def extract_numbers(input_string):
    pattern = r'<<(\d+)>>'
    numbers = re.findall(pattern, input_string)
    return [int(num) for num in numbers]

def main(args):
    pred_path = args.pred_path
    anno_path = args.anno_path
    use_lidartraj = args.use_lidartraj
    if use_lidartraj: # False
        sparsedrive_infos = pickle.load(open('data/infos/nuscenes_infos_val.pkl', 'rb'))
        sparsedrive_traj = {}
        for info in sparsedrive_infos['infos']:
            gt_planning = copy.deepcopy(info['gt_ego_fut_trajs'])[:,:2]
            gt_planning[:, 1] = -info['gt_ego_fut_trajs'][:, 0]
            gt_planning[:, 0] = info['gt_ego_fut_trajs'][:, 1]
            sparsedrive_traj[info['token']] = gt_planning.cumsum(axis=0)

    key_infos = pickle.load(open(osp.join(args.base_path, anno_path), 'rb'))
    preds = dict()
    
    # 初始化分类正确率统计
    classification_stats = {
        'total_samples': 0,
        'correct_predictions': 0,
        'available': False
    }
    
    for data in key_infos['infos']:
        if use_lidartraj:
            data['gt_planning'] = sparsedrive_traj[data['token']][None]
    

        if os.path.exists(os.path.join(pred_path, data['token']+'.pkl')):
            with open(os.path.join(pred_path, data['token']+'.pkl'),'rb')as f:
                pred_data = pickle.load(f).squeeze()
                preds[data['token']] = pred_data

        # 检查并加载分类正确率数据（如果启用了分类评估）
        if args.eval_classification:
            # 根据模型保存路径，分类结果保存在 vla_results 目录下
            # 尝试多个可能的路径位置
            possible_paths = [
                # 直接在pred_path目录下
                os.path.join(pred_path, data['token'] + '_classify_acc.txt'),
                # 在pred_path的上级目录的vla_results下
                os.path.join(os.path.dirname(pred_path.rstrip('/')), 'vla_results', data['token'] + '_classify_acc.txt'),
                # 在pred_path同级的vla_results目录下  
                os.path.join(os.path.dirname(pred_path.rstrip('/')) if pred_path.rstrip('/') else '.', 'vla_results', data['token'] + '_classify_acc.txt'),
            ]
            
            classify_acc_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    classify_acc_path = path
                    break
                    
            if classify_acc_path:
                try:
                    with open(classify_acc_path, 'r') as f:
                        acc_str = f.read().strip()
                        is_correct = 'true' in acc_str.lower()
                        classification_stats['total_samples'] += 1
                        if is_correct:
                            classification_stats['correct_predictions'] += 1
                        classification_stats['available'] = True
                except Exception as e:
                    print(f"Warning: Failed to read classification accuracy for {data['token']}: {e}")

    metric_dict = {
        'plan_obj_box_col_1s': 0,
        'plan_obj_box_col_2s': 0,
        'plan_obj_box_col_3s': 0,
        'plan_boundary_1s':0, 
        'plan_boundary_2s':0, 
        'plan_boundary_3s':0, 
        'l2_1s': 0,
        'l2_2s': 0,
        'l2_3s': 0,
        'samples':0,
    }

    num_threads = args.num_threads  
    total_data = len(key_infos['infos'])
    data_per_thread = total_data // num_threads
    threads = []
    lock = threading.Lock()
    pbar = tqdm(total=total_data)
    for i in range(num_threads):
        start = i * data_per_thread
        end = start + data_per_thread
        if i == num_threads - 1:
            end = total_data  
        thread = threading.Thread(target=process_data, args=(preds, start, end, key_infos, metric_dict, lock, pbar, planning_metric))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()
    pbar.close()    

    # 计算平均指标
    samples = metric_dict['samples']
    mean_plan_obj_box_col = sum([metric_dict[f'plan_obj_box_col_{i}s'] for i in range(1,4)]) / (3 * samples)
    mean_plan_boundary = sum([metric_dict[f'plan_boundary_{i}s'] for i in range(1,4)]) / (3 * samples)
    mean_l2 = sum([metric_dict[f'l2_{i}s'] for i in range(1,4)]) / (3 * samples)

    print(f"mean_plan_obj_box_col: {mean_plan_obj_box_col}")
    print(f"mean_plan_boundary: {mean_plan_boundary}")
    print(f"mean_l2: {mean_l2}")

    # 输出分类正确率结果
    if args.eval_classification:
        if classification_stats['available'] and classification_stats['total_samples'] > 0:
            classification_accuracy = classification_stats['correct_predictions'] / classification_stats['total_samples']
            print(f"Classification Accuracy: {classification_accuracy:.4f} ({classification_stats['correct_predictions']}/{classification_stats['total_samples']})")
        else:
            print("Classification accuracy data not available - no classification accuracy files found")
            print("Note: Classification files should be named as '{token}_classify_acc.txt' and contain 'True' or 'False'")

    for k in metric_dict:
        if k != "samples":
            print(f"{k}: {metric_dict[k]/samples}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate planning performance with optional classification accuracy.")
    parser.add_argument('--base_path', type=str, default='../data/nuscenes/', help='Base path to the data.')
    parser.add_argument('--pred_path', type=str, default='results_planning_only/', help='Path to the prediction results.')
    parser.add_argument('--anno_path', type=str, default='nuscenes2d_ego_temporal_infos_val.pkl', help='Path to the annotation file.')
    parser.add_argument('--num_threads', type=int, default=4, help='Number of threads to use.')
    parser.add_argument('--use_lidartraj', action='store_true', help='Whether to use lidartraj.')
    parser.add_argument('--eval_classification', action='store_true', 
                       help='Whether to evaluate classification accuracy. Requires *_classify_acc.txt files.')

    args = parser.parse_args()
    
    planning_metric = PlanningMetric(args.base_path)
    main(args)


