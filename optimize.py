import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import messagebox, ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# --- DİL AYARLARI VE SÖZLÜK ---
current_lang = "TR"

TEXTS = {
    "TR": {
        "title": "MAKİNE ÖĞRENMESİ TABANLI YILDIZ SINIFLANDIRMA",
        "tabs": ["Veri Girişi", "Cost Grafiği", "Karar Sınırı", "Regresyon", "Öğrenme Eğrisi", "Performans", "K-Means"],
        "input_title": "Yıldız Özellikleri",
        "inputs": ["Sıcaklık (K)", "Parlaklık (L)", "Yarıçap (R)", "Mutlak Mag (Mv)"],
        "btn_analyze": "ANALİZ ET",
        "btn_lang": "Language: TR",
        "res_wait": "Sonuç bekleniyor...",
        "giant": "SÜPER DEV",
        "dwarf": "CÜCE / ANA KOL",
        "prob": "Dev İhtimali",
        "group": "K-Means Grubu",
        "err_file": "stars.csv bulunamadı!",
        "err_num": "Lütfen sayısal değer giriniz.",
        "g_cost": ["Eğitim Hatası (Cost)", "İterasyon"],
        "g_dec": "Dev Yıldız Sınıflandırma Sınırı",
        "g_reg": "Sıcaklık - Büyüklük İlişkisi",
        "g_learn": "Öğrenme Eğrisi (Bias/Variance)",
        "g_conf": "Karmaşıklık Matrisi (Confusion Matrix)",
        "g_km": "K-Means Kümeleri",
        "cm_labels": ['Diğer', 'Dev']
    },
    "EN": {
        "title": "ML BASED STAR CLASSIFICATION SYSTEM",
        "tabs": ["Data Input", "Cost Graph", "Decision Boundary", "Regression", "Learning Curve", "Performance", "K-Means"],
        "input_title": "Star Properties",
        "inputs": ["Temperature (K)", "Luminosity (L)", "Radius (R)", "Absolute Mag (Mv)"],
        "btn_analyze": "ANALYZE",
        "btn_lang": "Dil: EN",
        "res_wait": "Waiting for result...",
        "giant": "SUPER GIANT",
        "dwarf": "DWARF / MAIN SEQ",
        "prob": "Giant Probability",
        "group": "K-Means Group",
        "err_file": "stars.csv not found!",
        "err_num": "Please enter numeric values.",
        "g_cost": ["Training Cost", "Iteration"],
        "g_dec": "Giant Star Classification Boundary",
        "g_reg": "Temp - Magnitude Relation",
        "g_learn": "Learning Curve (Bias/Variance)",
        "g_conf": "Confusion Matrix",
        "g_km": "K-Means Clusters",
        "cm_labels": ['Other', 'Giant']
    }
}

# --- VERİ YÜKLEME VE ÖN İŞLEME ---
try:
    df = pd.read_csv('stars.csv')
    # Özellikler: Temp, Luminosity, Radius, Abs Magnitude
    X_raw = df[['Temperature (K)', 'Luminosity(L/Lo)', 'Radius(R/Ro)', 'Absolute magnitude(Mv)']].values
    y = (df['Star type'] >= 4).astype(int).values # 1:Dev, 0:Cüce
except FileNotFoundError:
    messagebox.showerror("Error", "stars.csv bulunamadı! Lütfen dosyanın aynı klasörde olduğundan emin olun."); exit()

mu, sigma = np.mean(X_raw, axis=0), np.std(X_raw, axis=0)
X_sc = (X_raw - mu) / sigma
X_train = np.c_[np.ones(len(y)), X_sc] 

# --- MATEMATİKSEL İŞLEMLER ---
def sigmoid(z): return 1 / (1 + np.exp(-z))
def cost(h, y): return (-1/len(y)) * np.sum(y * np.log(np.clip(h, 1e-15, 1-1e-15)) + (1-y) * np.log(1-np.clip(h, 1e-15, 1-1e-15)))

# A) Gradient Descent (Lojistik Regresyon)
theta_log = np.zeros(X_train.shape[1])
cost_hist = []
for _ in range(2000):
    h = sigmoid(X_train @ theta_log)
    theta_log -= 0.01 * (X_train.T @ (h - y)) / len(y)
    cost_hist.append(cost(h, y))

# B) Doğrusal Regresyon (Analitik Çözüm)
X_lin = X_train[:, [0, 1]] # Bias ve Sıcaklık
y_lin = X_sc[:, 3]         # Mutlak Büyüklük
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

# D) Öğrenme Eğrisi Verileri
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

# --- ARAYÜZ VE GRAFİK YÖNETİMİ ---
root = tk.Tk()
root.geometry("1000x750")

# Dil Değiştirme Fonksiyonu
def change_language():
    global current_lang
    current_lang = "EN" if current_lang == "TR" else "TR"
    update_ui()

# UI Güncelleme Fonksiyonu
def update_ui():
    t = TEXTS[current_lang]
    
    # Ana Başlık ve Butonlar
    root.title(t["title"])
    btn_lang.config(text=t["btn_lang"])
    btn_analiz.config(text=t["btn_analyze"])
    lbl_input_title.config(text=t["input_title"])
    
    # Sekme İsimleri
    for i, tab_name in enumerate(t["tabs"]):
        nb.tab(i, text=tab_name)
    
    # Giriş Etiketleri
    for i, lbl in enumerate(input_labels_widgets):
        lbl.config(text=t["inputs"][i])
        
    # Sonuç Etiketi (Eğer bekliyorsa güncelle)
    current_res_text = lbl_res.cget("text")
    if current_res_text == "-" or current_res_text in [TEXTS["TR"]["res_wait"], TEXTS["EN"]["res_wait"]]:
        lbl_res.config(text=t["res_wait"])
    elif "Dev İhtimali" in current_res_text or "Giant Probability" in current_res_text:
        # Sonuç varsa onu da çevirmek için analizi tekrar tetikleyebiliriz veya metni parse edebiliriz.
        # Basitlik için kullanıcı tekrar "Analiz Et" diyebilir, ya da burayı boş bırakabiliriz.
        pass
        
    # Grafikleri Yenile
    plot_cost(axs[0]); canvases[0].draw()
    plot_sinir(axs[1]); canvases[1].draw()
    plot_reg(axs[2]); canvases[2].draw()
    plot_learn(axs[3]); canvases[3].draw()
    plot_conf(axs[4]); canvases[4].draw()
    plot_km(axs[5]); canvases[5].draw()

# Üst Bar (Dil Butonu için)
top_frame = tk.Frame(root)
top_frame.pack(side="top", fill="x", padx=10, pady=5)
btn_lang = tk.Button(top_frame, text=TEXTS["TR"]["btn_lang"], command=change_language, bg="#ddd")
btn_lang.pack(side="right")

nb = ttk.Notebook(root)
nb.pack(fill='both', expand=1)

# Tab'leri oluştur
tab_frames = []
for i in range(7):
    f = tk.Frame(nb)
    nb.add(f, text="") # İsimler update_ui ile gelecek
    tab_frames.append(f)

# --- GİRİŞ PANELİ (Tab 0) ---
f_L = tk.Frame(tab_frames[0], bg="white", padx=20)
f_L.pack(side='left', fill='both', expand=1)

# BURASI DÜZELTİLDİ:
f_R = tk.Frame(tab_frames[0], padx=20)
f_R.pack(side='right', fill='both', expand=1)

input_labels_widgets = []
entries = []

lbl_input_title = tk.Label(f_L, text="", font=("bold", 14), bg="white")
lbl_input_title.pack(pady=10)

defaults = ["5778", "1.0", "1.0", "4.83"]
for i, v in enumerate(defaults):
    l = tk.Label(f_L, text="", bg="white", anchor='w')
    l.pack(fill='x')
    input_labels_widgets.append(l) # Referansı sakla
    e = ttk.Entry(f_L); e.insert(0, v); e.pack(fill='x'); entries.append(e)

lbl_res = tk.Label(f_L, text="-", font=("Arial", 11), bg="white")
lbl_res.pack(pady=15)
txt_inf = tk.Text(f_R, height=20, bg="#fff3e0")
txt_inf.pack(fill='both', expand=1)

def analiz():
    try:
        t = TEXTS[current_lang]
        vals = np.array([float(e.get()) for e in entries])
        v_sc = (vals - mu) / sigma
        
        # 1. Lojistik Tahmin
        prob = sigmoid(np.dot(np.r_[1, v_sc], theta_log))
        # 2. K-Means Tahmin
        cl_pred = np.argmin([np.linalg.norm(v_sc - c) for c in centroids])
        
        status = t["giant"] if prob > 0.5 else t["dwarf"]
        lbl_res.config(
            text=f"{t['prob']}: %{prob*100:.1f}\n{status}\n{t['group']}: {cl_pred+1}", 
            fg="red" if prob>0.5 else "green"
        )
        
        txt_inf.delete(1.0, tk.END)
        for i, c in enumerate(centroids * sigma + mu):
            p = ">>" if i == cl_pred else "  "
            txt_inf.insert(tk.END, f"{p} {i+1}: T={c[0]:.0f}, Mag={c[3]:.2f}\n")
    except ValueError: 
        messagebox.showerror("Hata", TEXTS[current_lang]["err_num"])

btn_analiz = tk.Button(f_L, text="", command=analiz, bg="darkred", fg="white", font=("bold", 11))
btn_analiz.pack(fill='x', pady=10)

# --- GRAFİK ÇİZİMLERİ ---
axs = []
canvases = []

def cizim_alani(parent):
    fig = Figure(figsize=(5, 4), dpi=100)
    ax = fig.add_subplot(111)
    canvas = FigureCanvasTkAgg(fig, master=parent)
    canvas.get_tk_widget().pack(fill='both', expand=1)
    axs.append(ax)
    canvases.append(canvas)
    return ax

# 1. Cost Function
cizim_alani(tab_frames[1])
def plot_cost(ax):
    t = TEXTS[current_lang]
    ax.clear()
    ax.plot(cost_hist)
    ax.set_title(t["g_cost"][0])
    ax.set_xlabel(t["g_cost"][1])
    ax.grid(True, linestyle=':')

# 2. Karar Sınırı
cizim_alani(tab_frames[2])
def plot_sinir(ax):
    t = TEXTS[current_lang]
    ax.clear()
    ax.scatter(X_sc[:,0], X_sc[:,3], c=y, cmap='coolwarm', alpha=0.6)
    x_b = np.array([X_sc[:,0].min(), X_sc[:,0].max()])
    y_b = -(theta_log[0] + theta_log[1]*x_b) / theta_log[4] 
    ax.plot(x_b, y_b, 'k--', lw=2, label='Boundary')
    ax.invert_xaxis(); ax.invert_yaxis(); ax.legend()
    ax.set_title(t["g_dec"])
    ax.grid(True, linestyle=':')

# 3. Regresyon
cizim_alani(tab_frames[3])
def plot_reg(ax):
    t = TEXTS[current_lang]
    ax.clear()
    ax.scatter(X_sc[:,0], X_sc[:,3], alpha=0.4, label='Data')
    x_l = np.linspace(X_sc[:,0].min(), X_sc[:,0].max(), 100)
    ax.plot(x_l, theta_lin[0] + theta_lin[1]*x_l, 'r', lw=3, label='Reg')
    ax.invert_xaxis(); ax.invert_yaxis(); ax.legend()
    ax.set_title(t["g_reg"])
    ax.grid(True, linestyle=':')

# 4. Öğrenme Eğrisi
cizim_alani(tab_frames[4])
def plot_learn(ax):
    t = TEXTS[current_lang]
    ax.clear()
    ax.plot(sizes, tr_err, 'r-+', label='Train')
    ax.plot(sizes, val_err, 'b-', label='Validation') 
    ax.legend()
    ax.set_title(t["g_learn"])
    ax.grid(True, linestyle=':')

# 5. Confusion Matrix
cizim_alani(tab_frames[5])
def plot_conf(ax):
    t = TEXTS[current_lang]
    ax.clear()
    p = (sigmoid(X_train @ theta_log) > 0.5).astype(int)
    cm = [[np.sum((p==0)&(y==0)), np.sum((p==1)&(y==0))], [np.sum((p==0)&(y==1)), np.sum((p==1)&(y==1))]]
    ax.matshow(cm, cmap='Oranges', alpha=0.6)
    for (i,j), z in np.ndenumerate(cm): ax.text(j, i, z, ha='center', va='center', fontsize=20)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(t["cm_labels"])
    ax.set_yticklabels(t["cm_labels"])
    ax.set_title(t["g_conf"])

# 6. K-Means
cizim_alani(tab_frames[6])
def plot_km(ax):
    t = TEXTS[current_lang]
    ax.clear()
    ax.scatter(X_sc[:,0], X_sc[:,3], c=clusters, cmap='viridis', alpha=0.5)
    ax.scatter(centroids[:,0], centroids[:,3], c='r', marker='X', s=150)
    ax.invert_xaxis(); ax.invert_yaxis()
    ax.set_title(t["g_km"])
    ax.grid(True, linestyle=':')

# Başlangıçta arayüzü güncelle ve çalıştır
update_ui()
root.mainloop()