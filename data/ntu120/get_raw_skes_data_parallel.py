# Parallelized NTU120 skeleton parsing using multiprocessing
import os
import os.path as osp
import numpy as np
import pickle
import logging
from multiprocessing import Pool, cpu_count
from functools import partial


def parse_single_skeleton(ske_name, skes_path, skes_path_120):
    """Parse a single NTU .skeleton file. Returns dict or None."""
    # S018+ subjects are in the 120 extension set
    if int(ske_name[1:4]) >= 18:
        path = skes_path_120
    else:
        path = skes_path

    ske_file = osp.join(path, ske_name + '.skeleton')
    if not osp.exists(ske_file):
        return None

    try:
        with open(ske_file, 'r') as fr:
            lines = fr.readlines()
    except Exception:
        return None

    num_frames = int(lines[0].strip())
    bodies_data = {}
    frames_drop = []
    line_idx = 1
    valid_frames = -1

    for f in range(num_frames):
        if line_idx >= len(lines):
            break
        num_bodies = int(lines[line_idx].strip())
        line_idx += 1

        if num_bodies == 0:
            frames_drop.append(f)
            continue

        valid_frames += 1

        for b in range(num_bodies):
            if line_idx >= len(lines):
                break
            body_info = lines[line_idx].strip().split()
            body_id = body_info[0] if len(body_info) > 0 else str(b)
            line_idx += 1

            num_joints = int(lines[line_idx].strip())
            line_idx += 1

            joints = np.zeros((num_joints, 3), dtype=np.float32)
            colors = np.zeros((num_joints, 2), dtype=np.float32)

            for j in range(num_joints):
                if line_idx >= len(lines):
                    break
                joint_info = lines[line_idx].strip().split()
                line_idx += 1

                joints[j, 0] = float(joint_info[0])  # x
                joints[j, 1] = float(joint_info[1])  # y
                joints[j, 2] = float(joint_info[2])  # z
                colors[j, 0] = float(joint_info[5])  # colorX
                colors[j, 1] = float(joint_info[6])  # colorY

            if body_id not in bodies_data:
                bodies_data[body_id] = {
                    'joints': joints,
                    'colors': colors[np.newaxis],
                    'interval': [valid_frames],
                }
            else:
                bd = bodies_data[body_id]
                bd['joints'] = np.vstack((bd['joints'], joints))
                bd['colors'] = np.vstack((bd['colors'], colors[np.newaxis]))
                bd['interval'].append(bd['interval'][-1] + 1)

    num_valid = num_frames - len(frames_drop)
    if num_valid == 0:
        return None

    if len(bodies_data) > 1:
        for bd in bodies_data.values():
            bd['motion'] = np.sum(np.var(
                bd['joints'].reshape(-1, 25, 3), axis=0))

    return {
        'name': ske_name,
        'data': bodies_data,
        'num_frames': num_valid,
        'frames_drop': frames_drop,
    }


def main():
    save_path = './'
    skes_path = '../nturgbd_raw/nturgb+d_skeletons/'
    skes_path_120 = '../nturgbd_raw/nturgb+d_skeletons120/'
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

    parse_fn = partial(parse_single_skeleton,
                       skes_path=skes_path, skes_path_120=skes_path_120)

    raw_skes_data = []
    frames_cnt = np.zeros(num_files, dtype=int)
    frames_drop_skes = {}
    done = 0

    with Pool(n_workers) as pool:
        for idx, result in enumerate(pool.imap(parse_fn, skes_name, chunksize=64)):
            if result is not None:
                if result.get('frames_drop'):
                    frames_drop_skes[result['name']] = np.array(
                        result['frames_drop'], dtype=np.int32)
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
