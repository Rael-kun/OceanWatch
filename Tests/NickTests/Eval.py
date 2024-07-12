import Prog
import torch
import os
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
cf = Prog.Config()
model = Prog.TrAISformer(cf, partition_model=None)
## Evaluation
# ===============================
# Load the best model

# FIRST DATA
## Data
# ===============================
moving_threshold = 0.05
Data, aisdatasets = {}, {}
aisdls = {}
datapath = os.path.join(cf.datadir, cf.testset_name)
print(f"Loading {datapath}...")
with open(datapath, "rb") as f:
    l_pred_errors = pickle.load(f)
for V in l_pred_errors.values():
    try:
        moving_idx = np.where(V["traj"][:, 2] > moving_threshold)[0][0]
    except:
        moving_idx = len(V["traj"]) - 1  # This track will be removed
    V["traj"] = V["traj"][moving_idx:, :]
Data = [x for x in l_pred_errors.values() if not np.isnan(x["traj"]).any() and len(x["traj"]) > cf.min_seqlen]
print(len(l_pred_errors), len(Data))
print(f"Length: {len(Data)}")
print("Creating pytorch dataset...")
# Latter in this scipt, we will use inputs = x[:-1], targets = x[1:], hence
# max_seqlen = cf.max_seqlen + 1.
if cf.mode in ("pos_grad", "grad"):
    aisdatasets = Prog.AISDataset_grad(Data,
                                            max_seqlen=cf.max_seqlen + 1,
                                            device=cf.device)
else:
    aisdatasets = Prog.AISDataset(Data,
                                            max_seqlen=cf.max_seqlen + 1,
                                            device=cf.device)
    shuffle = True
    aisdls = DataLoader(aisdatasets,
                            batch_size=cf.batch_size,
                            shuffle=shuffle)
# NOW MODEL LOAD

print("Loading best model...")
#model.load_state_dict(torch.load(cf.ckpt_path))
path = ""
model.load_state_dict(torch.load(path))
print("Model Loaded")
v_ranges = torch.tensor([2, 3, 0, 0]).to(cf.device)
v_roi_min = torch.tensor([model.lat_min, -7, 0, 0]).to(cf.device)
max_seqlen = cf.init_seqlen + 6 * 4

model.eval()
print(cf.init_seqlen, max_seqlen)
l_min_errors, l_mean_errors, l_masks = [], [], []
pbar = tqdm(enumerate(aisdls), total=len(aisdls))
with torch.no_grad():
    for it, (seqs, masks, seqlens, mmsis, time_starts) in pbar:
        seqs_init = seqs[:, :cf.init_seqlen, :].to(cf.device)
        masks = masks[:, :max_seqlen].to(cf.device)
        batchsize = seqs.shape[0]
        error_ens = torch.zeros((batchsize, max_seqlen - cf.init_seqlen, cf.n_samples)).to(cf.device)
        for i_sample in range(cf.n_samples):
            preds = Prog.sample(model,
                                    seqs_init,
                                    max_seqlen - cf.init_seqlen,
                                    temperature=1.0,
                                    sample=True,
                                    sample_mode=cf.sample_mode,
                                    r_vicinity=cf.r_vicinity,
                                    top_k=cf.top_k)
            inputs = seqs[:, :max_seqlen, :].to(cf.device)
            input_coords = (inputs * v_ranges + v_roi_min) * torch.pi / 180
            pred_coords = (preds * v_ranges + v_roi_min) * torch.pi / 180
            d = Prog.haversine(input_coords, pred_coords) * masks
            error_ens[:, :, i_sample] = d[:, cf.init_seqlen:]

            # Plot time!
            for i in range(len(input_coords[..., :])):
                plt.plot(input_coords[..., 0], input_coords[..., 1])
                plt.plot(pred_coords[..., 0], pred_coords[..., 1])
                plt.savefig(cf.savedir + "indivPath.png")
        # Accumulation through batches
        l_min_errors.append(error_ens.min(dim=-1))
        l_mean_errors.append(error_ens.mean(dim=-1))
        l_masks.append(masks[:, cf.init_seqlen:])

l_min = [x.values for x in l_min_errors]
m_masks = torch.cat(l_masks, dim=0)
min_errors = torch.cat(l_min, dim=0) * m_masks
pred_errors = min_errors.sum(dim=0) / m_masks.sum(dim=0)
pred_errors = pred_errors.detach().cpu().numpy()

## Plot
# ===============================
plt.figure(figsize=(9, 6), dpi=150)
v_times = np.arange(len(pred_errors)) / 6
plt.plot(v_times, pred_errors)

timestep = 6
plt.plot(1, pred_errors[timestep], "o")
plt.plot([1, 1], [0, pred_errors[timestep]], "r")
plt.plot([0, 1], [pred_errors[timestep], pred_errors[timestep]], "r")
plt.text(1.12, pred_errors[timestep] - 0.5, "{:.4f}".format(pred_errors[timestep]), fontsize=10)

timestep = 12
plt.plot(2, pred_errors[timestep], "o")
plt.plot([2, 2], [0, pred_errors[timestep]], "r")
plt.plot([0, 2], [pred_errors[timestep], pred_errors[timestep]], "r")
plt.text(2.12, pred_errors[timestep] - 0.5, "{:.4f}".format(pred_errors[timestep]), fontsize=10)

timestep = 18
plt.plot(3, pred_errors[timestep], "o")
plt.plot([3, 3], [0, pred_errors[timestep]], "r")
plt.plot([0, 3], [pred_errors[timestep], pred_errors[timestep]], "r")
plt.text(3.12, pred_errors[timestep] - 0.5, "{:.4f}".format(pred_errors[timestep]), fontsize=10)
plt.xlabel("Time (hours)")
plt.ylabel("Prediction errors (km)")
plt.xlim([0, 12])
plt.ylim([0, 20])
# plt.ylim([0,pred_errors.max()+0.5])
plt.savefig(cf.savedir + "prediction_error.png")