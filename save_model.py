import torch

curr_path = '/home/durante/MOMA-SGG/output/6_8_2022_obj_detection/model_0045000.pth'
new_path = "/home/ssd/data/models/45k_67_model_final.pth"

original = torch.load(curr_path)

new = {"model": original["model"]}
torch.save(new, new_path)