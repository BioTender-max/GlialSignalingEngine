
import numpy as np
np.random.seed(42)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import os, shutil

OUT_DIR = '/mnt/shared-workspace/shared'
os.makedirs(OUT_DIR, exist_ok=True)

N_SAMPLES = 200
N_AD = 100
N_CTRL = 100
labels = np.array(['AD'] * N_AD + ['Control'] * N_CTRL)
is_ad = labels == 'AD'

# ── Astrocyte reactivity ──────────────────────────────────────────────────────
gfap_ad = np.random.lognormal(2.5, 0.5, N_AD)
gfap_ctrl = np.random.lognormal(1.5, 0.4, N_CTRL)
gfap = np.concatenate([gfap_ad, gfap_ctrl])

s100b_ad = np.random.lognormal(2.0, 0.4, N_AD)
s100b_ctrl = np.random.lognormal(1.2, 0.3, N_CTRL)
s100b = np.concatenate([s100b_ad, s100b_ctrl])

aqp4_ad = np.random.lognormal(1.8, 0.5, N_AD)
aqp4_ctrl = np.random.lognormal(1.5, 0.4, N_CTRL)
aqp4 = np.concatenate([aqp4_ad, aqp4_ctrl])

reactivity_score = (gfap + s100b + aqp4) / 3
t_react, p_react = stats.ttest_ind(reactivity_score[is_ad], reactivity_score[~is_ad])

# ── Microglial activation states ──────────────────────────────────────────────
# Homeostatic, DAM (disease-associated microglia), Inflammatory
mg_homeostatic_ad = np.random.beta(2, 5, N_AD)
mg_homeostatic_ctrl = np.random.beta(5, 2, N_CTRL)
mg_homeostatic = np.concatenate([mg_homeostatic_ad, mg_homeostatic_ctrl])

mg_dam_ad = np.random.beta(4, 3, N_AD)
mg_dam_ctrl = np.random.beta(2, 6, N_CTRL)
mg_dam = np.concatenate([mg_dam_ad, mg_dam_ctrl])

mg_inflam_ad = np.random.beta(3, 4, N_AD)
mg_inflam_ctrl = np.random.beta(2, 7, N_CTRL)
mg_inflam = np.concatenate([mg_inflam_ad, mg_inflam_ctrl])

# ── Cytokine network ──────────────────────────────────────────────────────────
cytokines = ['IL-1β', 'TNF-α', 'IL-6', 'IFN-γ', 'IL-10', 'TGF-β', 'IL-4', 'IL-12']
n_cyt = len(cytokines)
cyt_matrix_ad = np.random.lognormal(1.5, 0.5, (N_AD, n_cyt))
cyt_matrix_ctrl = np.random.lognormal(0.5, 0.3, (N_CTRL, n_cyt))
# Pro-inflammatory cytokines elevated in AD
for i, cyt in enumerate(cytokines[:4]):
    cyt_matrix_ad[:, i] *= 2.5
cyt_matrix = np.vstack([cyt_matrix_ad, cyt_matrix_ctrl])
cyt_corr = np.corrcoef(cyt_matrix.T)

# ── Neuroinflammation score ───────────────────────────────────────────────────
neuro_inflam = (cyt_matrix[:, :4].mean(axis=1) - cyt_matrix[:, 4:].mean(axis=1))
t_inflam, p_inflam = stats.ttest_ind(neuro_inflam[is_ad], neuro_inflam[~is_ad])

# ── Metabolic coupling (lactate shuttle) ─────────────────────────────────────
lactate_astro = np.random.lognormal(1.5, 0.4, N_SAMPLES)
lactate_neuron = 0.7 * lactate_astro + np.random.normal(0, 0.5, N_SAMPLES)
lactate_r, lactate_p = stats.pearsonr(lactate_astro, lactate_neuron)

# ── DAM signature ─────────────────────────────────────────────────────────────
dam_genes = ['TREM2', 'ApoE', 'Cst7', 'Lpl', 'Cd9', 'Clec7a', 'Itgax', 'Spp1']
dam_expr_ad = np.random.lognormal(2.0, 0.6, (N_AD, len(dam_genes)))
dam_expr_ctrl = np.random.lognormal(0.8, 0.4, (N_CTRL, len(dam_genes)))
dam_expr = np.vstack([dam_expr_ad, dam_expr_ctrl])
dam_score = dam_expr.mean(axis=1)

# ── Complement activation ─────────────────────────────────────────────────────
c1q_ad = np.random.lognormal(2.2, 0.5, N_AD)
c1q_ctrl = np.random.lognormal(1.0, 0.4, N_CTRL)
c3_ad = np.random.lognormal(2.0, 0.5, N_AD)
c3_ctrl = np.random.lognormal(1.1, 0.4, N_CTRL)
complement_score = np.concatenate([c1q_ad + c3_ad, c1q_ctrl + c3_ctrl])

# ── Synaptic pruning ──────────────────────────────────────────────────────────
pruning_ad = np.random.lognormal(1.8, 0.5, N_AD)
pruning_ctrl = np.random.lognormal(1.0, 0.4, N_CTRL)
pruning_score = np.concatenate([pruning_ad, pruning_ctrl])
t_prune, p_prune = stats.ttest_ind(pruning_score[is_ad], pruning_score[~is_ad])

# ── Dashboard ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 3, figsize=(20, 15))
fig.patch.set_facecolor('#0d1117')
fig.suptitle('Glial Signaling Engine — 200 Brain Samples (AD vs Control)',
             color='white', fontsize=16, fontweight='bold', y=0.98)

def style_ax(ax, title, xlabel='', ylabel=''):
    ax.set_facecolor('#161b22')
    ax.set_title(title, color='white', fontsize=11, fontweight='bold', pad=8)
    ax.set_xlabel(xlabel, color='#8b949e', fontsize=9)
    ax.set_ylabel(ylabel, color='#8b949e', fontsize=9)
    ax.tick_params(colors='#8b949e', labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')

# Panel 1: Astrocyte reactivity scores
ax = axes[0, 0]
ax.hist(reactivity_score[is_ad], bins=25, color='#f78166', alpha=0.7, label='AD', edgecolor='#0d1117')
ax.hist(reactivity_score[~is_ad], bins=25, color='#58a6ff', alpha=0.7, label='Control', edgecolor='#0d1117')
ax.legend(fontsize=8, facecolor='#21262d', labelcolor='white')
style_ax(ax, f'Astrocyte Reactivity (p={p_react:.2e})', 'Reactivity Score', 'Count')

# Panel 2: Microglial state distribution
ax = axes[0, 1]
states = ['Homeostatic', 'DAM', 'Inflammatory']
ad_means = [mg_homeostatic[is_ad].mean(), mg_dam[is_ad].mean(), mg_inflam[is_ad].mean()]
ctrl_means = [mg_homeostatic[~is_ad].mean(), mg_dam[~is_ad].mean(), mg_inflam[~is_ad].mean()]
x = np.arange(len(states))
w = 0.35
bars1 = ax.bar(x - w/2, ad_means, w, color='#f78166', alpha=0.85, label='AD', edgecolor='#0d1117')
bars2 = ax.bar(x + w/2, ctrl_means, w, color='#58a6ff', alpha=0.85, label='Control', edgecolor='#0d1117')
ax.set_xticks(x)
ax.set_xticklabels(states, color='#8b949e', fontsize=8)
ax.legend(fontsize=8, facecolor='#21262d', labelcolor='white')
style_ax(ax, 'Microglial Activation States', 'State', 'Mean Score')

# Panel 3: Cytokine network heatmap
ax = axes[0, 2]
im = ax.imshow(cyt_corr, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
ax.set_xticks(range(n_cyt))
ax.set_yticks(range(n_cyt))
ax.set_xticklabels(cytokines, rotation=45, ha='right', fontsize=7, color='#8b949e')
ax.set_yticklabels(cytokines, fontsize=7, color='#8b949e')
cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
cb.set_label('Pearson r', color='#8b949e', fontsize=8)
plt.setp(cb.ax.yaxis.get_ticklabels(), color='#8b949e', fontsize=7)
style_ax(ax, 'Cytokine Network Correlation', '', '')

# Panel 4: Neuroinflammation score
ax = axes[1, 0]
ax.hist(neuro_inflam[is_ad], bins=25, color='#f78166', alpha=0.7, label='AD', edgecolor='#0d1117')
ax.hist(neuro_inflam[~is_ad], bins=25, color='#58a6ff', alpha=0.7, label='Control', edgecolor='#0d1117')
ax.axvline(neuro_inflam[is_ad].mean(), color='#f78166', lw=2, linestyle='--')
ax.axvline(neuro_inflam[~is_ad].mean(), color='#58a6ff', lw=2, linestyle='--')
ax.legend(fontsize=8, facecolor='#21262d', labelcolor='white')
style_ax(ax, f'Neuroinflammation Score (p={p_inflam:.2e})', 'Score', 'Count')

# Panel 5: Metabolic coupling
ax = axes[1, 1]
ax.scatter(lactate_astro[is_ad], lactate_neuron[is_ad], color='#f78166', alpha=0.5, s=15,
           label='AD', edgecolors='none')
ax.scatter(lactate_astro[~is_ad], lactate_neuron[~is_ad], color='#58a6ff', alpha=0.5, s=15,
           label='Control', edgecolors='none')
m, b, r, p, _ = stats.linregress(lactate_astro, lactate_neuron)
xfit = np.linspace(lactate_astro.min(), lactate_astro.max(), 100)
ax.plot(xfit, m*xfit+b, color='#3fb950', lw=2, label=f'r={r:.3f}')
ax.legend(fontsize=8, facecolor='#21262d', labelcolor='white')
style_ax(ax, 'Glia-Neuron Metabolic Coupling (Lactate)', 'Astrocyte Lactate', 'Neuron Lactate')

# Panel 6: DAM signature
ax = axes[1, 2]
dam_ad_mean = dam_expr_ad.mean(axis=0)
dam_ctrl_mean = dam_expr_ctrl.mean(axis=0)
x_dam = np.arange(len(dam_genes))
ax.bar(x_dam - 0.2, dam_ad_mean, 0.4, color='#f78166', alpha=0.85, label='AD', edgecolor='#0d1117')
ax.bar(x_dam + 0.2, dam_ctrl_mean, 0.4, color='#58a6ff', alpha=0.85, label='Control', edgecolor='#0d1117')
ax.set_xticks(x_dam)
ax.set_xticklabels(dam_genes, rotation=45, ha='right', fontsize=7, color='#8b949e')
ax.legend(fontsize=8, facecolor='#21262d', labelcolor='white')
style_ax(ax, 'DAM Gene Signature', 'Gene', 'Expression')

# Panel 7: Complement activation
ax = axes[2, 0]
ax.hist(complement_score[is_ad], bins=25, color='#f78166', alpha=0.7, label='AD', edgecolor='#0d1117')
ax.hist(complement_score[~is_ad], bins=25, color='#58a6ff', alpha=0.7, label='Control', edgecolor='#0d1117')
t_comp, p_comp = stats.ttest_ind(complement_score[is_ad], complement_score[~is_ad])
ax.legend(fontsize=8, facecolor='#21262d', labelcolor='white')
style_ax(ax, f'Complement Activation (C1q+C3, p={p_comp:.2e})', 'Complement Score', 'Count')

# Panel 8: Synaptic pruning score
ax = axes[2, 1]
ax.hist(pruning_score[is_ad], bins=25, color='#f78166', alpha=0.7, label='AD', edgecolor='#0d1117')
ax.hist(pruning_score[~is_ad], bins=25, color='#58a6ff', alpha=0.7, label='Control', edgecolor='#0d1117')
ax.legend(fontsize=8, facecolor='#21262d', labelcolor='white')
style_ax(ax, f'Synaptic Pruning Score (p={p_prune:.2e})', 'Pruning Score', 'Count')

# Panel 9: Summary
ax = axes[2, 2]
ax.set_facecolor('#161b22')
ax.axis('off')
summary_text = (
    f"  Glial Signaling Summary\n"
    f"  {'─'*32}\n"
    f"  Brain samples:         {N_SAMPLES} (AD={N_AD}, Ctrl={N_CTRL})\n"
    f"  Astrocyte reactivity:\n"
    f"    AD mean:             {reactivity_score[is_ad].mean():.2f}\n"
    f"    Ctrl mean:           {reactivity_score[~is_ad].mean():.2f}\n"
    f"    p-value:             {p_react:.2e}\n"
    f"  Neuroinflammation:\n"
    f"    AD mean:             {neuro_inflam[is_ad].mean():.2f}\n"
    f"    Ctrl mean:           {neuro_inflam[~is_ad].mean():.2f}\n"
    f"    p-value:             {p_inflam:.2e}\n"
    f"  Lactate coupling r:    {lactate_r:.3f}\n"
    f"  Complement (AD/Ctrl):  {complement_score[is_ad].mean():.2f}/{complement_score[~is_ad].mean():.2f}\n"
    f"  Synaptic pruning p:    {p_prune:.2e}\n"
    f"  DAM score (AD/Ctrl):   {dam_score[is_ad].mean():.2f}/{dam_score[~is_ad].mean():.2f}\n"
)
ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', fontfamily='monospace',
        color='#e6edf3', bbox=dict(boxstyle='round', facecolor='#21262d', alpha=0.8))
ax.set_title('Summary Statistics', color='white', fontsize=11, fontweight='bold', pad=8)

plt.tight_layout(rect=[0, 0, 1, 0.97])
out_png = f'{OUT_DIR}/glial_signaling_engine_dashboard.png'
plt.savefig(out_png, dpi=100, bbox_inches='tight', facecolor='#0d1117')
plt.close()
print(f"Saved: {out_png}")

print("\n=== GlialSignalingEngine Key Results ===")
print(f"N samples: {N_SAMPLES} (AD={N_AD}, Control={N_CTRL})")
print(f"Astrocyte reactivity — AD: {reactivity_score[is_ad].mean():.3f}, Ctrl: {reactivity_score[~is_ad].mean():.3f}, p={p_react:.4e}")
print(f"Neuroinflammation — AD: {neuro_inflam[is_ad].mean():.3f}, Ctrl: {neuro_inflam[~is_ad].mean():.3f}, p={p_inflam:.4e}")
print(f"Lactate coupling correlation: r={lactate_r:.4f}, p={lactate_p:.4e}")
print(f"Complement score — AD: {complement_score[is_ad].mean():.3f}, Ctrl: {complement_score[~is_ad].mean():.3f}, p={p_comp:.4e}")
print(f"Synaptic pruning — AD: {pruning_score[is_ad].mean():.3f}, Ctrl: {pruning_score[~is_ad].mean():.3f}, p={p_prune:.4e}")
print(f"DAM score — AD: {dam_score[is_ad].mean():.3f}, Ctrl: {dam_score[~is_ad].mean():.3f}")
