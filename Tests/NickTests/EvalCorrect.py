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
# Test check
testAnomList = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 
                0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 
                1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 1, 1, 1]
#27
#[42, 68, 69, 72, 73, 76, 77, 79, 80, 81, 82, 83, 84, 86, 91, 92, 93, 94, 95, 97, 98, 99, 100, 101, 102, 104, 108, 109, 110, 111, 118, 119, 122, 123, 124, 126, 128, 129, 130, 131, 132, 133, 134, 135, 140, 141, 142, 143, 144, 145, 146, 147, 148, 154, 155, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 188, 189, 220, 233, 234, 283, 309, 310, 311, 312, 314, 315, 316, 319, 320, 322, 323, 330, 331, 332, 338, 339, 340, 341, 346, 347, 348, 349, 350, 351, 352, 353, 357, 358, 362, 363, 364, 375, 376, 378, 379, 393, 394, 484, 510]

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
#arr = []
#for i in range(521):
#    arr.append(i)
#for i in range(521):
#    # cycle through all l_pred_errors
#    x = l_pred_errors[i]
    #if (x in Data):
    #    print("Success")
    #else:
    #    print("Fail")

#    for j in range(len(Data)):
#        try:
#            if(x == Data[j]):
#                print(i)
#                arr.remove(i)
#        except:
#            continue
#print("Here is array", arr)
        
#Data = [x for x in l_pred_errors.values() if len(x["traj"]) > cf.min_seqlen]#if not np.isnan(x["traj"]).any()]
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
    shuffle = False
    aisdls = Prog.DataLoader(aisdatasets,
                            batch_size=1,#cf.batch_size,
                            shuffle=shuffle)
# NOW MODEL LOAD

print("Loading best model...")
#model.load_state_dict(torch.load(cf.ckpt_path))
path = "C:\\Users\\DSU\\Desktop\\AllCurrentClasses\\AISweden\\ProgAIS\\results\\Data-pos-pos_vicinity-10-40-blur-True-False-2-1.0-data_size-250-270-30-72-embd_size-256-256-128-128-head-4-4-bs-16-lr-0.0006-seqlen-18-120\\model_005.pt"
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
        if(testAnomList[it] == 1):
            # if an anomaly, skip
            continue
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
                plt.savefig(cf.savedir + "indivPathCorrect.png")
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
plt.savefig(cf.savedir + "prediction_errorCorrect.png")