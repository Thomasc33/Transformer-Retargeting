# Parallelized ETRI skeleton parsing using multiprocessing
# Drop-in replacement for get_raw_skes_data.py — same output format
import os
import os.path as osp
import numpy as np
import pickle
import logging
import csv
from multiprocessing import Pool, cpu_count
from functools import partial


def parse_single_skeleton(ske_name, skes_path):
    """Parse a single ETRI CSV skeleton file. Returns (bodies_data_dict, num_frames) or None."""
    ske_file = osp.join(skes_path, ske_name + '.csv')
    if not osp.exists(ske_file):
        parts = ske_name
        if len(parts) == 16 and '_' not in parts:
            ske_file = osp.join(skes_path,
                f"{parts[0:4]}_{parts[4:8]}_{parts[8:12]}_{parts[12:16]}.csv")
    if not osp.exists(ske_file):
        return None

    try:
        with open(ske_file, 'r') as fr:
            reader = csv.DictReader(fr)
            rows = list(reader)
    except Exception:
        return None

    if len(rows) == 0:
        return None

    joint_3d_cols = []
    joint_depth_cols = []
    for j in range(1, 26):
        joint_3d_cols.append((f'joint{j}_3dX', f'joint{j}_3dY', f'joint{j}_3dZ'))
        joint_depth_cols.append((f'joint{j}_depthX', f'joint{j}_depthY'))

    frames_by_num = {}
    for row in rows:
        try:
            frame_num = int(float(row.get('frameNum', 0)))
        except (ValueError, TypeError):
            continue
        if frame_num not in frames_by_num:
            frames_by_num[frame_num] = []
        frames_by_num[frame_num].append(row)

    sorted_frames = sorted(frames_by_num.keys())
    if len(sorted_frames) == 0:
        return None

    frames_drop = []
    bodies_data = {}
    valid_frames = -1

    for f_idx, frame_num in enumerate(sorted_frames):
        frame_rows = frames_by_num[frame_num]
        bodies_in_frame = {}
        for row in frame_rows:
            body_id = row.get('trackingID', row.get('bodyindexID', '0'))
            if body_id is None:
                body_id = '0'
            body_id = str(body_id).strip()
            if body_id not in bodies_in_frame:
                bodies_in_frame[body_id] = row

        if len(bodies_in_frame) == 0:
            frames_drop.append(f_idx)
            continue

        valid_frames += 1

        for bodyID, row in bodies_in_frame.items():
            joints_frame = np.zeros((25, 3), dtype=np.float32)
            colors_frame = np.zeros((25, 2), dtype=np.float32)

            for j in range(25):
                x_col, y_col, z_col = joint_3d_cols[j]
                dx_col, dy_col = joint_depth_cols[j]
                try:
                    x = float(row.get(x_col, 0) or 0)
                    y = float(row.get(y_col, 0) or 0)
                    z = float(row.get(z_col, 0) or 0)
                except (ValueError, TypeError):
                    x, y, z = 0.0, 0.0, 0.0
                try:
                    dx = float(row.get(dx_col, 0) or 0)
                    dy = float(row.get(dy_col, 0) or 0)
                except (ValueError, TypeError):
                    dx, dy = 0.0, 0.0

                if not np.isfinite(x): x = 0.0
                if not np.isfinite(y): y = 0.0
                if not np.isfinite(z): z = 0.0
                if not np.isfinite(dx): dx = 0.0
                if not np.isfinite(dy): dy = 0.0

                joints_frame[j] = [x, y, z]
                colors_frame[j] = [dx, dy]

            if bodyID not in bodies_data:
                body_data = {
                    'joints': joints_frame,
                    'colors': colors_frame[np.newaxis],
                    'interval': [valid_frames],
                }
            else:
                body_data = bodies_data[bodyID]
                body_data['joints'] = np.vstack((body_data['joints'], joints_frame))
                body_data['colors'] = np.vstack((body_data['colors'], colors_frame[np.newaxis]))
                body_data['interval'].append(body_data['interval'][-1] + 1)

            bodies_data[bodyID] = body_data

    num_valid = len(sorted_frames) - len(frames_drop)
    if num_valid == 0:
        return None

    if len(bodies_data) > 1:
        for body_data in bodies_data.values():
            body_data['motion'] = np.sum(np.var(
                body_data['joints'].reshape(-1, 25, 3), axis=0))

    return {
        'name': ske_name,
        'data': bodies_data,
        'num_frames': num_valid,
        'frames_drop': frames_drop,
    }


def main():
    save_path = './'
    skes_path = osp.join(save_path, 'csv')
    stat_path = osp.join(save_path, 'statistics')
    os.makedirs('./raw_data', exist_ok=True)

    skes_name_file = osp.join(stat_path, 'skes_available_name.txt')
    save_data_pkl = osp.join(save_path, 'raw_data', 'raw_skes_data.pkl')
    frames_drop_pkl = osp.join(save_path, 'raw_data', 'frames_drop_skes.pkl')

    skes_name = np.loadtxt(skes_name_file, dtype=str)
    num_files = skes_name.size
    print(f'Found {num_files} available skeleton files.')

    n_workers = min(cpu_count(), 128)
    print(f'Using {n_workers} workers')

    parse_fn = partial(parse_single_skeleton, skes_path=skes_path)

    raw_skes_data = []
    frames_cnt = np.zeros(num_files, dtype=np.int32)
    frames_drop_skes = {}
    done = 0

    with Pool(n_workers) as pool:
        for idx, result in enumerate(pool.imap(parse_fn, skes_name, chunksize=64)):
            if result is not None:
                if result.get('frames_drop'):
                    frames_drop_skes[result['name']] = np.array(
                        result['frames_drop'], dtype=np.int32)
                    del result['frames_drop']
                else:
                    if 'frames_drop' in result:
                        del result['frames_drop']
                raw_skes_data.append(result)
                frames_cnt[idx] = result['num_frames']

            done += 1
            if done % 5000 == 0:
                print(f'Processed: {100.0 * done / num_files:.1f}% ({done} / {num_files})')

    with open(save_data_pkl, 'wb') as fw:
        pickle.dump(raw_skes_data, fw, pickle.HIGHEST_PROTOCOL)
    np.savetxt(osp.join(save_path, 'raw_data', 'frames_cnt.txt'), frames_cnt, fmt='%d')

    print(f'Saved {len(raw_skes_data)} raw bodies data into {save_data_pkl}')
    print(f'Total frames: {np.sum(frames_cnt)}')

    with open(frames_drop_pkl, 'wb') as fw:
        pickle.dump(frames_drop_skes, fw, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
