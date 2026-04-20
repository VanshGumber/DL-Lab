import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("Electric_Production.csv", parse_dates=["DATE"], dayfirst=True)
df = df.sort_values("DATE").reset_index(drop=True)
vals = df["Value"].values.reshape(-1, 1)

sc = MinMaxScaler()
sv = sc.fit_transform(vals)

SEQ, PRED, EP, BS, LR = 24, 120, 80, 32, 1e-3
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def mk_seq(d, sl):
    X, y = [], []
    for i in range(len(d) - sl):
        X.append(d[i:i+sl]); y.append(d[i+sl])
    return np.array(X), np.array(y)

X, y = mk_seq(sv, SEQ)
sp = int(len(X) * 0.85)
Xtr, Xte = X[:sp], X[sp:]
ytr, yte = y[:sp], y[sp:]

dl_tr = DataLoader(TensorDataset(torch.tensor(Xtr, dtype=torch.float32),
                                  torch.tensor(ytr, dtype=torch.float32)),
                   batch_size=BS, shuffle=True)
dl_te = DataLoader(TensorDataset(torch.tensor(Xte, dtype=torch.float32),
                                  torch.tensor(yte, dtype=torch.float32)),
                   batch_size=BS, shuffle=False)

class RNNm(nn.Module):
    def __init__(self, h=64, nl=2):
        super().__init__()
        self.rnn = nn.RNN(1, h, nl, batch_first=True, dropout=0.2)
        self.fc  = nn.Linear(h, 1)
    def forward(self, x):
        o, _ = self.rnn(x); return self.fc(o[:, -1])

class LSTMm(nn.Module):
    def __init__(self, h=64, nl=2):
        super().__init__()
        self.lstm = nn.LSTM(1, h, nl, batch_first=True, dropout=0.2)
        self.fc   = nn.Linear(h, 1)
    def forward(self, x):
        o, _ = self.lstm(x); return self.fc(o[:, -1])

class GRUm(nn.Module):
    def __init__(self, h=64, nl=2):
        super().__init__()
        self.gru = nn.GRU(1, h, nl, batch_first=True, dropout=0.2)
        self.fc  = nn.Linear(h, 1)
    def forward(self, x):
        o, _ = self.gru(x); return self.fc(o[:, -1])

class PatchEmbed(nn.Module):
    def __init__(self, ps, d):
        super().__init__()
        self.ps = ps
        self.proj = nn.Linear(ps, d)
    def forward(self, x):
        B, T, _ = x.shape
        x = x.squeeze(-1)
        np_ = T // self.ps
        x = x[:, :np_*self.ps].reshape(B, np_, self.ps)
        return self.proj(x)

class ViTm(nn.Module):
    def __init__(self, sl=SEQ, ps=4, d=64, nh=4, nl=2):
        super().__init__()
        self.pe  = PatchEmbed(ps, d)
        np_      = sl // ps
        self.pos = nn.Parameter(torch.zeros(1, np_, d))
        enc      = nn.TransformerEncoderLayer(d, nh, d*2, dropout=0.1, batch_first=True)
        self.tr  = nn.TransformerEncoder(enc, nl)
        self.fc  = nn.Linear(d, 1)
    def forward(self, x):
        x = self.pe(x) + self.pos
        x = self.tr(x)
        return self.fc(x.mean(1))

def train(mdl, ep=EP):
    mdl.to(DEV)
    opt  = torch.optim.Adam(mdl.parameters(), lr=LR)
    sch  = torch.optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.5)
    crit = nn.MSELoss()
    hist = []
    for e in range(ep):
        mdl.train(); tl = 0
        for xb, yb in dl_tr:
            xb, yb = xb.to(DEV), yb.to(DEV)
            opt.zero_grad()
            loss = crit(mdl(xb), yb)
            loss.backward(); opt.step()
            tl += loss.item()
        sch.step()
        hist.append(tl / len(dl_tr))
        if (e+1) % 20 == 0:
            print(f"  ep {e+1}/{ep}  loss={hist[-1]:.5f}")
    return hist

def evaluate(mdl):
    mdl.eval()
    preds, acts = [], []
    with torch.no_grad():
        for xb, yb in dl_te:
            p = mdl(xb.to(DEV)).cpu().numpy()
            preds.extend(p.flatten())
            acts.extend(yb.numpy().flatten())
    pa = sc.inverse_transform(np.array(preds).reshape(-1,1)).flatten()
    aa = sc.inverse_transform(np.array(acts).reshape(-1,1)).flatten()
    mae  = mean_absolute_error(aa, pa)
    rmse = np.sqrt(mean_squared_error(aa, pa))
    mape = np.mean(np.abs((aa - pa) / aa)) * 100
    return mae, rmse, mape

def future_fc(mdl, n=PRED):
    mdl.eval()
    buf = sv[-SEQ:].flatten().tolist()
    out = []
    with torch.no_grad():
        for _ in range(n):
            x = torch.tensor(buf[-SEQ:], dtype=torch.float32).reshape(1, SEQ, 1).to(DEV)
            p = mdl(x).cpu().item()
            out.append(p); buf.append(p)
    return sc.inverse_transform(np.array(out).reshape(-1,1)).flatten()

mdls = {"RNN": RNNm(), "LSTM": LSTMm(), "GRU": GRUm(), "ViT": ViTm()}
res  = {}

for nm, m in mdls.items():
    print(f"\n── {nm} ──")
    hist            = train(m)
    mae, rmse, mape = evaluate(m)
    fc              = future_fc(m)
    res[nm]         = dict(hist=hist, fc=fc, mae=mae, rmse=rmse, mape=mape)
    print(f"  MAE={mae:.3f}  RMSE={rmse:.3f}  MAPE={mape:.2f}%")

last_dt  = df["DATE"].max()
fut_dts  = pd.date_range(last_dt + pd.DateOffset(months=1), periods=PRED, freq="MS")
fc_end   = fut_dts[-1]

COLS = {"RNN":"red","LSTM":"green","GRU":"blue","ViT":"purple"}

fig, ax = plt.subplots(figsize=(12, 6), facecolor="#0f1117")
fig.suptitle("Electric Production Forecast — RNN · LSTM · GRU · Vision Transformer",
             fontsize=16, color="white", y=1.01, fontweight="bold")

ax.set_facecolor("#1a1d27")
for nm, r in res.items():
    ax.plot(fut_dts, r["fc"], color=COLS[nm], lw=2.0, label=nm, alpha=0.9)
ax.axvline(last_dt, color="white", ls="--", lw=0.8, alpha=0.6)
ax.set_xlim(fut_dts[0] - pd.DateOffset(months=1), fc_end + pd.DateOffset(months=1))
ax.set_title(f"10-Year Forecast  {fut_dts[0].strftime('%b %Y')} → {fc_end.strftime('%b %Y')}",
             color="white", fontsize=13)
ax.legend(loc="upper left", fontsize=9, framealpha=0.2, labelcolor="white", ncol=4)
ax.tick_params(colors="white"); ax.spines[:].set_color("#333")
ax.set_ylabel("Production", color="white")

plt.savefig("forecast_output.png", dpi=150, bbox_inches="tight", facecolor="#0f1117")
plt.show()
print("\nSaved → forecast_output.png")

print(f"\n{'Model':<8} {'MAE':>8} {'RMSE':>8} {'MAPE%':>8}")
print("-"*36)
for nm, r in res.items():
    print(f"{nm:<8} {r['mae']:>8.3f} {r['rmse']:>8.3f} {r['mape']:>7.2f}%")
