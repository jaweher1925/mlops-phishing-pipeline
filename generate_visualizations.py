import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import re
import os
from collections import Counter

# Create visualizations folder
os.makedirs("visualizations", exist_ok=True)

# Load data
df = pd.read_csv("data/raw/AVN_Basic.csv")
df['urls_clean'] = df['urls'].apply(lambda x: int(str(x).strip()) if str(x).strip().isdigit() else 0)
df['subject_len'] = df['subject'].fillna('').apply(len)
df['body_len'] = df['body'].fillna('').apply(len)

COLORS = ['#1565C0', '#C62828', '#757575']

# ── Plot 1: Class Distribution ─────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
labels = ['Legitimate', 'Phishing', 'Garbage']
counts = [31122, 28476, 402]
bars = ax.bar(labels, counts, color=COLORS, edgecolor='white', linewidth=0.5)
ax.set_title('Class Distribution', fontweight='bold', fontsize=13)
ax.set_ylabel('Number of emails')
for bar, count in zip(bars, counts):
    pct = count / sum(counts) * 100
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
            f'{count:,}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_ylim(0, 36000)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations/01_class_distribution.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: 01_class_distribution.png")




# ── Plot 2: URL Count per Class ────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
url_avgs = [4.66, 5.53]
url_labels = ['Legitimate', 'Phishing']
bars2 = ax.bar(url_labels, url_avgs, color=['#1565C0', '#C62828'], edgecolor='white', width=0.5)
ax.set_title('Average URL Count per Email', fontweight='bold', fontsize=13)
ax.set_ylabel('Average number of URLs')
for bar, val in zip(bars2, url_avgs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f'{val:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_ylim(0, 7)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations/02_url_count.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: 02_url_count.png")




# ── Plot 3: Body Length Distribution ──────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
legit_body = df[df['label']==0]['body_len'].clip(0, 5000)
phish_body = df[df['label']==1]['body_len'].clip(0, 5000)
ax.hist(legit_body, bins=50, alpha=0.6, color='#1565C0', label='Legitimate', density=True)
ax.hist(phish_body, bins=50, alpha=0.6, color='#C62828', label='Phishing', density=True)
ax.set_title('Email Body Length Distribution', fontweight='bold', fontsize=13)
ax.set_xlabel('Body length (characters, clipped at 5000)')
ax.set_ylabel('Density')
ax.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations/03_body_length.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: 03_body_length.png")





# ── Plot 4: Subject Length Distribution ───────────────────
fig, ax = plt.subplots(figsize=(7, 5))
legit_subj = df[df['label']==0]['subject_len'].clip(0, 150)
phish_subj = df[df['label']==1]['subject_len'].clip(0, 150)
ax.hist(legit_subj, bins=40, alpha=0.6, color='#1565C0', label=f'Legitimate (avg: 34.7)', density=True)
ax.hist(phish_subj, bins=40, alpha=0.6, color='#C62828', label=f'Phishing (avg: 29.8)', density=True)
ax.set_title('Email Subject Length Distribution', fontweight='bold', fontsize=13)
ax.set_xlabel('Subject length (characters)')
ax.set_ylabel('Density')
ax.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations/04_subject_length.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: 04_subject_length.png")




# ── Plot 5: Top Keywords ───────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
phish_kws = ['your', 'daily', 'replica', 'alert', 'watches', 'custom']
phish_counts_kw = [3101, 2938, 2083, 1426, 1486, 1416]
legit_kws = ['python', 'opensuse', 'users', 'perl', 'spambayes', 'ilug']
legit_counts_kw = [4543, 2487, 599, 719, 544, 1165]
x = np.arange(len(phish_kws))
width = 0.35
ax.bar(x - width/2, phish_counts_kw, width, label='Phishing keywords', color='#C62828', alpha=0.8)
ax.bar(x + width/2, legit_counts_kw, width, label='Legitimate keywords', color='#1565C0', alpha=0.8)
ax.set_title('Top Subject Keywords by Class', fontweight='bold', fontsize=13)
ax.set_ylabel('Frequency')
ax.set_xticks(x)
ax.set_xticklabels([f'{p}\nvs\n{l}' for p, l in zip(phish_kws, legit_kws)], fontsize=8)
ax.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations/05_keywords.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: 05_keywords.png")




# ── Plot 6: Model Comparison ───────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
models = ['Logistic\nRegression', 'Random\nForest', 'AdaBoost']
f1_scores = [0.9822, 0.9440, 0.9138]
colors_m = ['#2E7D32', '#1565C0', '#757575']
bars3 = ax.bar(models, f1_scores, color=colors_m, edgecolor='white')
ax.set_title('Model F1 Score Comparison', fontweight='bold', fontsize=13)
ax.set_ylabel('Weighted F1 Score')
ax.set_ylim(0.85, 1.0)
for bar, val in zip(bars3, f1_scores):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
bars3[0].set_edgecolor('#1B5E20')
bars3[0].set_linewidth(2)
ax.annotate('Champion', xy=(0, 0.9822), xytext=(0.4, 0.975),
            fontsize=10, color='#2E7D32', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#2E7D32'))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations/06_model_comparison.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: 06_model_comparison.png")




# ── Plot 7: All EDA Combined ───────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle('AVN Phishing Email Dataset — Exploratory Data Analysis',
             fontsize=14, fontweight='bold', y=0.98)

# class dist
ax = axes[0,0]
bars = ax.bar(['Legitimate', 'Phishing', 'Garbage'], [31122, 28476, 402], color=COLORS, edgecolor='white')
ax.set_title('Class Distribution', fontweight='bold')
ax.set_ylabel('Count')
for bar, c in zip(bars, [31122, 28476, 402]):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+200,
            f'{c:,}\n({c/60000*100:.1f}%)', ha='center', va='bottom', fontsize=8, fontweight='bold')
ax.set_ylim(0,36000); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False); ax.grid(axis='y',alpha=0.3)

# url
ax = axes[0,1]
b2 = ax.bar(['Legitimate','Phishing'],[4.66,5.53],color=['#1565C0','#C62828'],width=0.5,edgecolor='white')
ax.set_title('Avg URL Count', fontweight='bold'); ax.set_ylabel('Avg URLs')
for b,v in zip(b2,[4.66,5.53]): ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.05,f'{v:.2f}',ha='center',va='bottom',fontsize=10,fontweight='bold')
ax.set_ylim(0,7); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False); ax.grid(axis='y',alpha=0.3)

# body len
ax = axes[1,0]
ax.hist(df[df['label']==0]['body_len'].clip(0,5000),bins=50,alpha=0.6,color='#1565C0',label='Legitimate',density=True)
ax.hist(df[df['label']==1]['body_len'].clip(0,5000),bins=50,alpha=0.6,color='#C62828',label='Phishing',density=True)
ax.set_title('Body Length Distribution', fontweight='bold'); ax.set_xlabel('Characters (max 5000)'); ax.set_ylabel('Density'); ax.legend(fontsize=8)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False); ax.grid(axis='y',alpha=0.3)

# keywords
ax = axes[1,1]
x2 = np.arange(6); w = 0.35
ax.bar(x2-w/2,[3101,2938,2083,1426,1486,1416],w,label='Phishing',color='#C62828',alpha=0.8)
ax.bar(x2+w/2,[4543,2487,599,719,544,1165],w,label='Legitimate',color='#1565C0',alpha=0.8)
ax.set_title('Top Subject Keywords', fontweight='bold'); ax.set_ylabel('Frequency')
ax.set_xticks(x2); ax.set_xticklabels(['your','daily','replica','alert','watches','custom'],rotation=20,ha='right',fontsize=8)
ax.legend(fontsize=8); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False); ax.grid(axis='y',alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/07_eda_combined.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: 07_eda_combined.png")

print("\nAll visualizations saved to visualizations/ folder!")
print("Files created:")
for f in sorted(os.listdir('visualizations')):
    print(f"  - {f}")