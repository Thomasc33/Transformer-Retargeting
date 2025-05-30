# Detailed Model Performance Metrics

| model_name | eval_model | action_recognition_accuracy | reidentification_accuracy | original_actor_accuracy | mse_gt | mse_reference | bone_length_consistency | joint_angle_limits | temporal_smoothness | velocity_consistency | foot_contact_consistency | gender_classification_orig | gender_classification_ret | gender_classification_cross | privacy_utility_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Raw Skeleton (No Anonymization) | sgn | 90.21 | 1.20 | 81.28 | 0.04 | 0.00 | 0.02 | 96.10 | 0.03 | 1.00 | 100.00 | 97.66 | 53.16 | 97.14 | 89.01 |
| Pose Motion Retargeting (PMR) | sgn | 17.80 | 6.41 | 17.14 | 0.04 | 0.01 | 0.03 | 92.53 | 0.06 | 0.02 | 73.32 | 80.69 | 52.59 | 79.64 | 11.39 |
| Deep Motion Retargeting (DMR) | sgn | 32.83 | 6.36 | 22.91 | 0.04 | 0.00 | 0.02 | 97.08 | 0.01 | 0.09 | 82.69 | 80.71 | 53.74 | 77.98 | 26.46 |
| Raw Skeleton (No Anonymization) | mixformer | 80.09 | 0.95 | 84.09 | 0.04 | 0.00 | 0.02 | 95.93 | 0.03 | 1.00 | 100.00 | 98.09 | 52.69 | 97.79 | 79.14 |
| Pose Motion Retargeting (PMR) | mixformer | 18.93 | 6.56 | 14.18 | 0.04 | 0.01 | 0.03 | 92.59 | 0.06 | 0.02 | 73.06 | 78.29 | 49.94 | 79.78 | 12.36 |
| Deep Motion Retargeting (DMR) | mixformer | 28.86 | 6.42 | 18.69 | 0.04 | 0.00 | 0.02 | 97.15 | 0.01 | 0.09 | 82.72 | 80.61 | 55.41 | 77.75 | 22.44 |
