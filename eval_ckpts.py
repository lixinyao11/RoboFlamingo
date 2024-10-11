import os
from datetime import datetime

ckpt_dir = 'RobotFlamingoDBG'

# ckpt_names = ['checkpoint_gripper_post_hist_1_aug_10_4_traj_cons_ws_12_mpt_9b_4.pth', 'checkpoint_gripper_post_hist_1_aug_10_4_traj_cons_ws_12_mpt_9b_3.pth', 'checkpoint_gripper_post_hist_1_aug_10_4_traj_cons_ws_12_mpt_9b_2.pth']
ckpt_names = ['checkpoint_gripper_post_hist_1_aug_10_4_traj_cons_ws_12_mpt_3b_4.pth']
print(ckpt_names)
student_ckpt = '/Share/xyli/logs/flamingo/ckpts/test_run_gpt_ddp/model_epoch_49.ckpt'
for ckpt_name in ckpt_names:
    use_gripper = 1 if 'gripper' in ckpt_name else 0
    use_state = 1 if 'state' in ckpt_name else 0
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    os.makedirs('logs_new', exist_ok=True)
    current_time = current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    # log_file = 'logs_new/evaluate_{}.log'.format(ckpt_name.split('.')[0])
    log_file = f'logs_new/evaluate_{current_time}.log'
    ckpt_ix = ckpt_names.index(ckpt_name)
    print('evaluating {}/{} checkpoint'.format(ckpt_ix+1, len(ckpt_names)))
    fusion_mode = 'pre'
    if 'post' in ckpt_name:
        fusion_mode = 'post'
    if 'two_way' in ckpt_name:
        fusion_mode = 'two_way'
    window_size = 8
    ckpt_attrs = ckpt_name.split('_')
    if 'ws' in ckpt_attrs:
        window_size = int(ckpt_attrs[ckpt_attrs.index('ws')+1])
    # print('bash robot_flamingo/pt_eval_ckpts.bash {} {} {} {}'.format(ckpt_path, log_file, use_gripper, use_state))
    # exit(0)
    node_num = 8
    if 'mpt_9b' in ckpt_name:
        node_num = 5
    os.system('bash robot_flamingo/pt_eval_ckpts.bash {} {} {} {} {} {} {} {}'.format(ckpt_path, log_file, use_gripper, use_state, fusion_mode, window_size, node_num, student_ckpt))

