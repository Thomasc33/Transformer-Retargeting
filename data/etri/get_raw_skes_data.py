# Adapted for ETRI Kinect v2 CSV format
# Original NTU version: Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import os.path as osp
import os
import numpy as np
import pickle
import logging
import csv


def get_raw_bodies_data(skes_path, ske_name, frames_drop_skes, frames_drop_logger):
    """
    Get raw bodies data from an ETRI Kinect v2 CSV skeleton sequence.

    ETRI CSV format (253 columns):
        frameNum, bodyindexID, trackingID,
        joint1_3dX, joint1_3dY, joint1_3dZ,
        joint1_depthX, joint1_depthY,
        joint1_orientationX/Y/Z/W, joint1_trackingState,
        ... (10 cols per joint x 25 joints)

    Each body's data is a dict that contains the following keys:
      - joints: raw 3D joints positions. Shape: (num_frames x 25, 3)
      - colors: raw 2D depth locations. Shape: (num_frames, 25, 2)
      - interval: a list which stores the frame indices of this body.
      - motion: motion amount (only for the sequence with 2 or more bodyIDs).

    Return:
      a dict for a skeleton sequence with 3 key-value pairs:
        - name: the skeleton filename.
        - data: a dict which stores raw data of each body.
        - num_frames: the number of valid frames.
    """
    # Try both naming conventions
    ske_file = osp.join(skes_path, ske_name + '.csv')
    if not osp.exists(ske_file):
        # Try with underscores: A001_P001_G001_C001.csv
        parts = ske_name
        if len(parts) == 16 and '_' not in parts:
            ske_file = osp.join(skes_path,
                f"{parts[0:4]}_{parts[4:8]}_{parts[8:12]}_{parts[12:16]}.csv")
    if not osp.exists(ske_file):
        print('Error: CSV file not found for %s' % ske_name)
        return None

    try:
        with open(ske_file, 'r') as fr:
            reader = csv.DictReader(fr)
            rows = list(reader)
    except Exception as e:
        print('Error reading %s: %s' % (ske_name, e))
        return None

    if len(rows) == 0:
        print('Error: Empty CSV file %s' % ske_name)
        return None

    # Build joint column names
    joint_3d_cols = []
    joint_depth_cols = []
    for j in range(1, 26):
        joint_3d_cols.append((f'joint{j}_3dX', f'joint{j}_3dY', f'joint{j}_3dZ'))
        joint_depth_cols.append((f'joint{j}_depthX', f'joint{j}_depthY'))

    # Group rows by frame number
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
    num_frames_total = len(sorted_frames)

    if num_frames_total == 0:
        print('Error: No valid frames in %s' % ske_name)
        return None

    frames_drop = []
    bodies_data = dict()
    valid_frames = -1

    for f_idx, frame_num in enumerate(sorted_frames):
        frame_rows = frames_by_num[frame_num]

        # Group by trackingID (or bodyindexID as fallback)
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
            # Extract 3D joints
            joints_frame = np.zeros((25, 3), dtype=np.float32)
            colors_frame = np.zeros((25, 2), dtype=np.float32)
            valid_joint = True

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

                # Replace inf/nan with 0
                if not np.isfinite(x): x = 0.0
                if not np.isfinite(y): y = 0.0
                if not np.isfinite(z): z = 0.0
                if not np.isfinite(dx): dx = 0.0
                if not np.isfinite(dy): dy = 0.0

                joints_frame[j] = [x, y, z]
                colors_frame[j] = [dx, dy]

            if bodyID not in bodies_data:
                body_data = dict()
                body_data['joints'] = joints_frame  # (25, 3)
                body_data['colors'] = colors_frame[np.newaxis]  # (1, 25, 2)
                body_data['interval'] = [valid_frames]
            else:
                body_data = bodies_data[bodyID]
                body_data['joints'] = np.vstack((body_data['joints'], joints_frame))
                body_data['colors'] = np.vstack((body_data['colors'], colors_frame[np.newaxis]))
                pre_frame_idx = body_data['interval'][-1]
                body_data['interval'].append(pre_frame_idx + 1)

            bodies_data[bodyID] = body_data

    num_frames_drop = len(frames_drop)
    num_valid = num_frames_total - num_frames_drop

    if num_valid == 0:
        print('Error: All frames data of %s is missing' % ske_name)
        return None

    if num_frames_drop > 0:
        frames_drop_skes[ske_name] = np.array(frames_drop, dtype=np.int32)
        frames_drop_logger.info('{}: {} frames missed: {}\n'.format(
            ske_name, num_frames_drop, frames_drop))

    # Calculate motion (only for the sequence with 2 or more bodyIDs)
    if len(bodies_data) > 1:
        for body_data in bodies_data.values():
            body_data['motion'] = np.sum(np.var(
                body_data['joints'].reshape(-1, 25, 3), axis=0))

    return {'name': ske_name, 'data': bodies_data, 'num_frames': num_valid}


def get_raw_skes_data():
    skes_name = np.loadtxt(skes_name_file, dtype=str)

    num_files = skes_name.size
    print('Found %d available skeleton files.' % num_files)

    raw_skes_data = []
    frames_cnt = np.zeros(num_files, dtype=np.int32)

    for (idx, ske_name) in enumerate(skes_name):
        bodies_data = get_raw_bodies_data(skes_path, ske_name, frames_drop_skes, frames_drop_logger)
        if bodies_data is None:
            continue
        raw_skes_data.append(bodies_data)
        frames_cnt[idx] = bodies_data['num_frames']
        if (idx + 1) % 1000 == 0:
            print('Processed: %.2f%% (%d / %d)' % \
                  (100.0 * (idx + 1) / num_files, idx + 1, num_files))

    with open(save_data_pkl, 'wb') as fw:
        pickle.dump(raw_skes_data, fw, pickle.HIGHEST_PROTOCOL)
    np.savetxt(osp.join(save_path, 'raw_data', 'frames_cnt.txt'), frames_cnt, fmt='%d')

    print('Saved raw bodies data into %s' % save_data_pkl)
    print('Total frames: %d' % np.sum(frames_cnt))

    with open(frames_drop_pkl, 'wb') as fw:
        pickle.dump(frames_drop_skes, fw, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    save_path = './'

    skes_path = osp.join(save_path, 'csv')
    stat_path = osp.join(save_path, 'statistics')
    if not osp.exists('./raw_data'):
        os.makedirs('./raw_data')

    skes_name_file = osp.join(stat_path, 'skes_available_name.txt')
    save_data_pkl = osp.join(save_path, 'raw_data', 'raw_skes_data.pkl')
    frames_drop_pkl = osp.join(save_path, 'raw_data', 'frames_drop_skes.pkl')

    frames_drop_logger = logging.getLogger('frames_drop')
    frames_drop_logger.setLevel(logging.INFO)
    frames_drop_logger.addHandler(logging.FileHandler(osp.join(save_path, 'raw_data', 'frames_drop.log')))
    frames_drop_skes = dict()

    get_raw_skes_data()

    with open(frames_drop_pkl, 'wb') as fw:
        pickle.dump(frames_drop_skes, fw, pickle.HIGHEST_PROTOCOL)
