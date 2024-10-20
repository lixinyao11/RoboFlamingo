import os
import numpy as np
import sys
sys.path.append(os.curdir)
from robot_flamingo.eval.eval_utils import get_cast_dtype, ModelWrapper

def process_calvin_data_ddp(args, wrapped_model, dataset_path, image_processor, tokenizer, future_act_len=-1, student_model=None):
    cast_dtype = get_cast_dtype(args.precision)
    hist_len = None
    if args.head_type=="diffusion":
        hist_len = args.n_obs_steps
    elif args.pad_length != -1:
        hist_len = args.pad_length
    wrapped_model = ModelWrapper(
        wrapped_model, 
        tokenizer, 
        image_processor, 
        student_model, 
        cast_dtype, 
        args.head_type=="diffusion", 
        history_len=hist_len, 
        future_act_len=future_act_len, 
        collect_data=True,
    )

    # data_dir = '/Share/xyli/calvin/dataset/task_D_D/training'
    # data_dir = '/home/xyli/Code/calvin/dataset/calvin_debug_dataset/training'
    data_dir = os.path.join(dataset_path, 'training')

    # one annot for one subtask, steps from start-to-end (epid)
    annots = os.path.join(data_dir, 'lang_annotations', 'auto_lang_ann.npy')
    annots  = np.load(annots, allow_pickle=True).item()
    task_num = len(annots['info']['indx'])
    print("task num: ", task_num)
    for i in range(task_num):
        if i % 2 != args.rank:
            continue
        print(f"task {i}")
        (start, end) = annots['info']['indx'][i]
        annot = annots['language']['ann'][i]
        annot = annot.split('\n')[0]
        if '\u2019' in annot:
            annot.replace('\u2019', '\'')
        
        wrapped_model.reset()
        wrapped_model.logger.start_rollout(i, annot)
        print(annot)

        for step in range(start, end):
            # turn 358482 into '0358482'
            step_str = str(step).zfill(7)
            data_path = os.path.join(data_dir, f'episode_{step_str}.npz')
            with np.load(data_path, allow_pickle=True) as data:
                if wrapped_model.replan != -1 and step % wrapped_model.replan == 0:
                    if wrapped_model.model.module.refresh != -1:
                        wrapped_model.model.module.lang_encoder.lm_head.hidden_state = None
                        wrapped_model.model.module.lang_encoder.lm_head.history_memory = wrapped_model.model.module.lang_encoder.lm_head.history_memory[-model.refresh:]
                    else:
                        wrapped_model.reset()
                wrapped_model.step(data, annot, process_data=True)

        wrapped_model.logger.end_rollout(0)
        wrapped_model.logger.save_rollout(f'episode_{step_str}')


