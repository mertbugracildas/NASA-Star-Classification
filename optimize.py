import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import messagebox, ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

#VERİ YÜKLEME VE ÖN İŞLEME
try:
    df = pd.read_csv('stars.csv')
    # Özellikler:
    X_raw = df[['Temperature (K)', 'Luminosity(L/Lo)', 'Radius(R/Ro)', 'Absolute magnitude(Mv)']].values
    y = (df['Star type'] >= 4).astype(int).values # 1:Dev, 0:Cüce
except FileNotFoundError:
    messagebox.showerror("Hata", "stars.csv bulunamadı!"); exit()

# Normalizasyon
mu, sigma = np.mean(X_raw, axis=0), np.std(X_raw, axis=0)
X_sc = (X_raw - mu) / sigma
X_train = np.c_[np.ones(len(y)), X_sc] 

# 2. MATEMATİKSEL İŞLEMLER
def sigmoid(z): return 1 / (1 + np.exp(-z))
def cost(h, y): return (-1/len(y)) * np.sum(y * np.log(np.clip(h, 1e-15, 1-1e-15)) + (1-y) * np.log(1-np.clip(h, 1e-15, 1-1e-15)))

# A) Gradient Descent
theta_log = np.zeros(X_train.shape[1])
cost_hist = []
for _ in range(2000): # İterasyon
    h = sigmoid(X_train @ theta_log)
    theta_log -= 0.01 * (X_train.T @ (h - y)) / len(y)
    cost_hist.append(cost(h, y))

# B) Doğrusal Regresyon 
# Sıcaklık - Mutlak Büyüklük ilişkisi
X_lin = X_train[:, [0, 1]]
y_lin = X_sc[:, 3]
theta_lin = np.linalg.inv(X_lin.T @ X_lin) @ X_lin.T @ y_lin

# C) K-Means 
def run_kmeans(K=6, iter=50):
    np.random.seed(42)
    ct = X_sc[np.random.choice(len(X_sc), K, replace=False)]
    cl = np.zeros(len(X_sc))
    for _ in range(iter):
        cl = np.argmin([np.linalg.norm(X_sc - c, axis=1) for c in ct], axis=0)
        new_ct = np.array([X_sc[cl == k].mean(axis=0) for k in range(K)])
        if np.all(ct == new_ct): break
        ct = new_ct
    return ct, cl
centroids, clusters = run_kmeans()

# D) Öğrenme Eğrisi
np.random.seed(1)
idx = np.random.permutation(len(y))
split = int(len(y)*0.8)
X_tr_full, y_tr_full = X_train[idx[:split]], y[idx[:split]]
X_val_fixed, y_val_fixed = X_train[idx[split:]], y[idx[split:]]
sizes, tr_err, val_err = range(10, len(X_tr_full), 10), [], []

for m in sizes:
    th_sub = np.zeros(X_train.shape[1])
    for _ in range(500): 
        th_sub -= 0.01 * (X_tr_full[:m].T @ (sigmoid(X_tr_full[:m] @ th_sub) - y_tr_full[:m])) / m
    tr_err.append(cost(sigmoid(X_tr_full[:m] @ th_sub), y_tr_full[:m]))
    val_err.append(cost(sigmoid(X_val_fixed @ th_sub), y_val_fixed))

#ARAYÜZ VE GRAFİK YÖNETİMİ
root = tk.Tk(); root.geometry("1000x720"); root.title("MAKİNE ÖĞRENMESİ TABANLI YILDIZ SINIFLANDIRMA VE ANALİZ SİSTEMİ")
nb = ttk.Notebook(root); nb.pack(fill='both', expand=1)

def cizim_yap(parent, func):
    fig = Figure(figsize=(5, 4), dpi=100); ax = fig.add_subplot(111)
    func(ax); ax.grid(True, linestyle=':')
    FigureCanvasTkAgg(fig, master=parent).get_tk_widget().pack(fill='both', expand=1)

tabs = {n: tk.Frame(nb) for n in ["Veri Girişi", "Cost Grafiği", "Karar Sınırı", "Regresyon", "Öğrenme Eğrisi", "Performans", "K-Means"]}
for n, f in tabs.items(): nb.add(f, text=n)

# Giriş Paneli
f_L = tk.Frame(tabs["Veri Girişi"], bg="white", padx=20); f_L.pack(side='left', fill='both', expand=1)
f_R = tk.Frame(tabs["Veri Girişi"], padx=20); f_R.pack(side='right', fill='both', expand=1)
entries = []
tk.Label(f_L, text="Yıldız Özellikleri", font=("bold", 14), bg="white").pack(pady=10)
for l, v in zip(["Sıcaklık (K)", "Parlaklık (L)", "Yarıçap (R)", "Mutlak Mag (Mv)"], ["5778", "1.0", "1.0", "4.83"]):
    tk.Label(f_L, text=l, bg="white", anchor='w').pack(fill='x')
    e = ttk.Entry(f_L); e.insert(0, v); e.pack(fill='x'); entries.append(e)

lbl_res = tk.Label(f_L, text="-", font=("Arial", 11), bg="white"); lbl_res.pack(pady=15)
txt_inf = tk.Text(f_R, height=20, bg="#fff3e0"); txt_inf.pack(fill='both', expand=1)

def analiz():
    try:
        vals = np.array([float(e.get()) for e in entries])
        v_sc = (vals - mu) / sigma
        # 1. Lojistik Tahmin
        prob = sigmoid(np.dot(np.r_[1, v_sc], theta_log))
        # 2. K-Means Tahmin
        cl_pred = np.argmin([np.linalg.norm(v_sc - c) for c in centroids])
        
        lbl_res.config(text=f"Dev İhtimali: %{prob*100:.1f}\n{'SÜPER DEV' if prob>0.5 else 'CÜCE / ANA KOL'}\nK-Means Grubu: {cl_pred+1}", fg="red" if prob>0.5 else "green")
        
        txt_inf.delete(1.0, tk.END)
        for i, c in enumerate(centroids * sigma + mu):
            p = ">>" if i == cl_pred else "  "
            txt_inf.insert(tk.END, f"{p} Küme {i+1}: T={c[0]:.0f}, Mag={c[3]:.2f}\n")
    except: messagebox.showerror("Hata", "Sayısal giriniz.")

tk.Button(f_L, text="ANALİZ ET", command=analiz, bg="darkred", fg="white", font=("bold", 11)).pack(fill='x', pady=10)

# 4. GRAFİK ÇİZİMLERİ

# 1. Cost Function
cizim_yap(tabs["Cost Grafiği"], lambda ax: [ax.plot(cost_hist), ax.set_title("Eğitim Hatası (Cost)"), ax.set_xlabel("İterasyon")])

# 2. Karar Sınırı
def plot_sinir(ax):
    ax.scatter(X_sc[:,0], X_sc[:,3], c=y, cmap='coolwarm', alpha=0.6)
    x_b = np.array([X_sc[:,0].min(), X_sc[:,0].max()])
    y_b = -(theta_log[0] + theta_log[1]*x_b) / theta_log[4] 
    ax.plot(x_b, y_b, 'k--', lw=2, label='Sınır')
    ax.invert_xaxis(); ax.invert_yaxis(); ax.legend(); ax.set_title("Dev Yıldız Sınıflandırma Sınırı")
cizim_yap(tabs["Karar Sınırı"], plot_sinir)

# 3. Doğrusal Regresyon
def plot_reg(ax):
    ax.scatter(X_sc[:,0], X_sc[:,3], alpha=0.4, label='Veri')
    x_l = np.linspace(X_sc[:,0].min(), X_sc[:,0].max(), 100)
    ax.plot(x_l, theta_lin[0] + theta_lin[1]*x_l, 'r', lw=3, label='Regresyon')
    ax.invert_xaxis(); ax.invert_yaxis(); ax.legend(); ax.set_title("Sıcaklık - Büyüklük İlişkisi")
cizim_yap(tabs["Regresyon"], plot_reg)

# 4. Öğrenme Eğrisi
cizim_yap(tabs["Öğrenme Eğrisi"], lambda ax: [
    ax.plot(sizes, tr_err, 'r-+', label='Train'), 
    ax.plot(sizes, val_err, 'b-', label='Validation'), 
    ax.legend(), ax.set_title("Öğrenme Eğrisi (Bias/Variance)")
])

# 5. Confusion Matrix
def plot_conf(ax):
    p = (sigmoid(X_train @ theta_log) > 0.5).astype(int)
    cm = [[np.sum((p==0)&(y==0)), np.sum((p==1)&(y==0))], [np.sum((p==0)&(y==1)), np.sum((p==1)&(y==1))]]
    ax.matshow(cm, cmap='Oranges', alpha=0.6)
    for (i,j), z in np.ndenumerate(cm): ax.text(j, i, z, ha='center', va='center', fontsize=20)
    

    ax.set_xticks([0, 1]) 
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Diğer', 'Dev'])
    ax.set_yticklabels(['Diğer', 'Dev'])

    
    ax.set_title("Confusion Matrix")
cizim_yap(tabs["Performans"], plot_conf)

# 6. K-Means
def plot_km(ax):
    ax.scatter(X_sc[:,0], X_sc[:,3], c=clusters, cmap='viridis', alpha=0.5)
    ax.scatter(centroids[:,0], centroids[:,3], c='r', marker='X', s=150)
    ax.invert_xaxis(); ax.invert_yaxis(); ax.set_title("K-Means Kümeleri")
cizim_yap(tabs["K-Means"], plot_km)

analiz()
root.mainloop()