import os
import h5py
import numpy as np
from datetime import datetime

class EvaluationLogger:
    def __init__(self, output_dir="/Share/xyli/Datasets/flamingo_data/D", save_data=False):
        self.enabled = save_data
        if not self.enabled:
            return
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f'{output_dir}_{current_time}'
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.sequence_id = 0
        self.subtask_id = 0
        self.subtask_txt = None
        self.step_cnt = 0
        self.current_sequence_data = []
        self.current_subtask_data = []


    def start_sequence(self):
        if not self.enabled:
            return
        self.sequence_id += 1
        self.current_sequence_data = []  # 每个 sequence 开始时清空数据


    def save_sequence(self):
        if not self.enabled:
            return
        print("end sequence", self.sequence_id)
        # return
        file_name = f"sequence_{self.sequence_id}.hdf5"
        file_path = os.path.join(self.output_dir, file_name)
        with h5py.File(file_path, "w") as f:
            for subtask in self.current_sequence_data:
                group = f.create_group(f"subtask_{subtask['subtask_id']}")
                for key, value in subtask.items():
                    if key == "sequence_id" or key == "subtask_id" or key == "success" or key == "steps":
                        group.attrs[key] = value
                    else:
                        group.create_dataset(key, data=value)
        print(f"Saved sequence {self.sequence_id} to {file_path}")
        subtask_data = self.current_sequence_data[0]
        for key in subtask_data.keys():
            if key == "sequence_id" or key == "subtask_id" or key == "success" or key == "steps" or key == "lang":
                print(key, subtask_data[key])
            else:
                print(key, subtask_data[key].shape)


    def start_rollout(self, subtask_id, text):
        if not self.enabled:
            return
        self.subtask_id = subtask_id
        self.subtask_txt = text
        self.step_cnt = 0
        self.current_subtask_data = []  

    
    def save_rollout(self, file_name):
        if not self.enabled:
            return
        # save as npz
        file_path = os.path.join(self.output_dir, f'{file_name}.npz')
        np.savez(file_path, **self.current_subtask_data)
        self.current_sequence_data = []


    def end_rollout(self, success):
        if not self.enabled:
            return
        subtask_data = {
            # "sequence_id": self.sequence_id,
            "subtask_id": self.subtask_id,
            # "success": success,
            "steps": self.step_cnt,
            "lang": self.subtask_txt
        }
        keys = self.current_subtask_data[0].keys()
        subtask_data.update({key: [] for key in keys})
        for step in self.current_subtask_data:
            for key in keys:
                subtask_data[key].append(step[key])
        for key in keys:
            subtask_data[key] = np.stack(subtask_data[key])
        self.current_sequence_data.append(subtask_data)
        self.current_subtask_data = subtask_data


    def log_step(self, image, cls, state, text, mask, action, feature):
        if not self.enabled:
            return
        step_data = {
            "image": image,
            "image.pooled": cls,
            "state": state,
            "text": text,
            # "mask": mask,
            "action.logits": action,
            # "action.logits": action[0][0],  # (7,)
            "features": feature,
            # "features": feature[0],  # (13, 2048)
        }
        self.current_subtask_data.append(step_data)
        self.step_cnt += 1
