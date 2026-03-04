

# --------------------------------------------------------------------------"""
Conceptual framework diagram — 8-step U-shaped flowchart (FancyBboxPatch + annotate arrows). Phase colours: data=blue, analysis=teal, model=purple, eval=orange, output=green. Saves source_attribution_framework.png.
"""


# ============================================================
# CONCEPTUAL FRAMEWORK — Microbial Source Attribution Pipeline
# Visualises the 8-step analytical workflow as a flowchart
# ============================================================
import matplotlib.patches as mpatches
import matplotlib.patheffects as mpe

PHASE_COLOURS = {
    'data':    '#1565C0',   # blue  — data preparation
    'analysis':'#00796B',   # teal  — statistical analysis
    'model':   '#6A1B9A',   # purple — machine learning
    'eval':    '#E65100',   # orange — validation
    'output':  '#2E7D32',   # green  — results
}

# ── Step definitions ─────────────────────────────────────────────────────────
steps = [
    # Row 1 (top): data → analysis
    dict(x=0.5,  y=0.75, w=1.3, h=0.30, phase='data',
         title='Input Data',
         body='Species (CLR)\nMetabolites (log2)'),
    dict(x=2.2,  y=0.75, w=1.5, h=0.30, phase='analysis',
         title='Stage-Stratified Corr.',
         body='Spearman ρ  (q<0.05, ρ>0)\nper disease stage'),
    dict(x=4.1,  y=0.75, w=1.5, h=0.30, phase='analysis',
         title='Source Label Derivation',
         body='argmax CLR among\ncorrelated species'),
    dict(x=6.0,  y=0.75, w=1.5, h=0.30, phase='model',
         title='Feature Matrix',
         body='log2 metabolites\n+ study group dummies'),
    # Row 2 (bottom): model → output (right-to-left to form a U-shape)
    dict(x=6.0,  y=0.30, w=1.5, h=0.30, phase='model',
         title='RF Pre-screen',
         body='Top 50 features\nby importance'),
    dict(x=4.1,  y=0.30, w=1.5, h=0.30, phase='model',
         title='Multi-Model CV',
         body='RF · GB · LR · XGB\n10×3 stratified folds'),
    dict(x=2.2,  y=0.30, w=1.5, h=0.30, phase='eval',
         title='Benchmark',
         body='CV metrics +\nliterature concordance'),
    dict(x=0.5,  y=0.30, w=1.3, h=0.30, phase='output',
         title='Attribution',
         body='Top source species\nper polyamine'),
]

fig, ax = plt.subplots(figsize=(14, 5))
ax.set_xlim(0, 8)
ax.set_ylim(0, 1.1)
ax.axis('off')
ax.set_title('Conceptual Framework: Microbial Source Attribution in Metabolomics',
             fontsize=13, fontweight='bold', pad=10)

# Draw boxes
for s in steps:
    col = PHASE_COLOURS[s['phase']]
    rect = mpatches.FancyBboxPatch(
        (s['x'], s['y']), s['w'], s['h'],
        boxstyle='round,pad=0.02',
        facecolor=col + '22',        # light fill (hex alpha)
        edgecolor=col, linewidth=2)
    ax.add_patch(rect)
    cx = s['x'] + s['w'] / 2
    cy = s['y'] + s['h'] / 2
    ax.text(cx, cy + 0.055, s['title'],
            ha='center', va='center', fontsize=8.5, fontweight='bold',
            color=col)
    ax.text(cx, cy - 0.055, s['body'],
            ha='center', va='center', fontsize=7.5, color='#333333',
            linespacing=1.4)

# ── Arrows row 1 (left → right) ──────────────────────────────────────────────
for i in range(3):
    x0 = steps[i]['x'] + steps[i]['w']
    x1 = steps[i + 1]['x']
    y  = steps[i]['y'] + steps[i]['h'] / 2
    ax.annotate('', xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle='->', color='#555', lw=1.5))

# ── Down arrow: step 3 → step 4 (feature matrix down to pre-screen) ──────────
x_down = steps[3]['x'] + steps[3]['w'] / 2
ax.annotate('', xy=(x_down, steps[4]['y'] + steps[4]['h']),
            xytext=(x_down, steps[3]['y']),
            arrowprops=dict(arrowstyle='->', color='#555', lw=1.5))

# ── Arrows row 2 (right → left) ──────────────────────────────────────────────
for i in range(4, 7):
    x0 = steps[i]['x']
    x1 = steps[i + 1]['x'] + steps[i + 1]['w']
    y  = steps[i]['y'] + steps[i]['h'] / 2
    ax.annotate('', xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle='->', color='#555', lw=1.5))

# ── Legend ────────────────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(facecolor=PHASE_COLOURS['data'] + '44',
                   edgecolor=PHASE_COLOURS['data'], label='Data preparation'),
    mpatches.Patch(facecolor=PHASE_COLOURS['analysis'] + '44',
                   edgecolor=PHASE_COLOURS['analysis'], label='Statistical analysis'),
    mpatches.Patch(facecolor=PHASE_COLOURS['model'] + '44',
                   edgecolor=PHASE_COLOURS['model'], label='Machine learning'),
    mpatches.Patch(facecolor=PHASE_COLOURS['eval'] + '44',
                   edgecolor=PHASE_COLOURS['eval'], label='Validation'),
    mpatches.Patch(facecolor=PHASE_COLOURS['output'] + '44',
                   edgecolor=PHASE_COLOURS['output'], label='Application / output'),
]
ax.legend(handles=legend_items, loc='lower center', bbox_to_anchor=(0.5, -0.12),
          ncol=5, fontsize=8.5, frameon=False)

plt.tight_layout()
fw_path = CLF_FIG_DIR / 'source_attribution_framework.png'
plt.savefig(fw_path, dpi=150, bbox_inches='tight')
plt.show()
plt.close()
print(f"Saved: source_attribution_framework.png")
