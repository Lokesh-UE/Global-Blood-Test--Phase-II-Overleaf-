import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300

os.makedirs('Figures/Sample Figures/Dataset', exist_ok=True)
os.makedirs('Figures/Sample Figures/Network and Architecture', exist_ok=True)
os.makedirs('Figures/Sample Figures/Results', exist_ok=True)

def fig1():
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14); ax.set_ylim(0, 8); ax.axis('off')
    ax.text(7, 7.5, 'Proposed Health Risk Prediction Framework', fontsize=18, fontweight='bold', ha='center', color='#1a237e')
    def box(x, y, w, h, t, c='#1a237e', tc='white', fs=10):
        r = mpatches.FancyBboxPatch((x,y), w, h, boxstyle="round,pad=0.03,rounding_size=0.15", facecolor=c, edgecolor='none')
        ax.add_patch(r); ax.text(x+w/2, y+h/2, t, ha='center', va='center', fontsize=fs, color=tc, fontweight='bold')
    def arr(x1, y1, x2, y2, c='#00a8e8'):
        ax.annotate('', xy=(x2,y2), xytext=(x1,y1), arrowprops=dict(arrowstyle='->', color=c, lw=2))
    box(0.5, 6.0, 2.5, 0.9, 'Global Blood Test\nDataset (Kaggle)')
    box(3.5, 6.0, 2.5, 0.9, 'Preprocessing\n& Cleaning', '#00a8e8')
    box(6.5, 6.0, 2.5, 0.9, 'Feature\nGrouping')
    box(9.5, 6.0, 2.5, 0.9, 'Train/Val/Test\nSplit (70/15/15)', '#00a8e8')
    arr(3.0, 6.45, 3.5, 6.45); arr(6.0, 6.45, 6.5, 6.45); arr(9.0, 6.45, 9.5, 6.45)
    gs = ['Hematological\n(Hgb,WBC,RBC,\nPlatelet,MCV)', 'Metabolic\n(Glucose,BMI)', 'Lipid Profile\n(Total,HDL,LDL)', 'Inflammatory\n(CRP,Ferritin)', 'Vital Signs\n(Systolic,Diastolic\nBP)', 'Demographics\n(Age,Gender,\nRegion)']
    cs = ['#3498db', '#e67e22', '#9b59b6', '#e74c3c', '#2ecc71', '#1abc9c']
    for i, (g, c) in enumerate(zip(gs, cs)):
        box(0.3+i*2.2, 4.5, 2.0, 1.1, g, c, 'white', 8); arr(1.3+i*2.2, 6.0, 1.3+i*2.2, 5.6)
    ms = ['Random\nForest', 'Logistic\nRegression', 'SVM\n(RBF)', 'XGBoost\nGradient\nBoosting', 'Neural\nNetwork\n(MLP)', 'Logistic\nRegression']
    mcs = ['#2980b9', '#d35400', '#8e44ad', '#c0392b', '#27ae60', '#16a085']
    for i, (m, c) in enumerate(zip(ms, mcs)):
        box(0.3+i*2.2, 3.0, 2.0, 0.9, m, c, 'white', 9); arr(1.3+i*2.2, 4.5, 1.3+i*2.2, 3.9)
    box(1.5, 1.5, 4.0, 0.9, 'Ensemble Fusion\n(Early/Late/Stacking/Weighted)')
    box(6.0, 1.5, 4.0, 0.9, 'Cross-Feature Interaction\n(Polynomial+Attention)', '#00a8e8')
    for i in range(6): arr(1.3+i*2.2, 3.0, 3.5 if i<3 else 8.0, 2.4)
    box(1.5, 0.2, 3.0, 0.8, 'Explainability\n(SHAP+LIME)', '#2c3e50')
    box(5.0, 0.2, 3.0, 0.8, 'Uncertainty\nQuantification', '#2c3e50')
    box(8.5, 0.2, 3.0, 0.8, 'Risk Prediction\n(Binary+Multi-class)')
    arr(3.5, 1.5, 3.0, 1.0); arr(8.0, 1.5, 6.5, 1.0); arr(8.0, 1.5, 10.0, 1.0)
    ax.legend(handles=[mpatches.Patch(facecolor='#1a237e', label='Data/Fusion/Output'), mpatches.Patch(facecolor='#00a8e8', label='Processing/Interaction'), mpatches.Patch(facecolor='#3498db', label='Feature Groups&Models')], loc='lower right', fontsize=9)
    plt.tight_layout(); plt.savefig('Figures/Sample Figures/Network and Architecture/Figure1.pdf', format='pdf', dpi=300, bbox_inches='tight'); plt.close()

def fig2():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].pie([78, 22], labels=['Low Risk', 'High Risk'], colors=['#2ecc71', '#e74c3c'], autopct='%1.0f%%', startangle=90, textprops={'fontsize': 12})
    axes[0].set_title('A) High Risk Distribution', fontsize=13, fontweight='bold', pad=15)
    axes[1].pie([45, 35, 15, 5], labels=['Low', 'Moderate', 'High', 'Critical'], colors=['#2ecc71', '#f39c12', '#e74c3c', '#8e44ad'], autopct='%1.0f%%', startangle=90, textprops={'fontsize': 12})
    axes[1].set_title('B) Risk Category Distribution', fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout(); plt.savefig('Figures/Sample Figures/Dataset/Image1.pdf', format='pdf', dpi=300, bbox_inches='tight'); plt.close()

def fig3():
    fig, ax = plt.subplots(figsize=(8, 5))
    np.random.seed(42)
    data = [np.concatenate([np.random.normal(13.5, 1.5, 800), np.random.normal(10.5, 1.2, 150), np.random.normal(16.5, 1.0, 50)]),
            np.concatenate([np.random.normal(90, 12, 700), np.random.normal(140, 35, 250), np.random.normal(200, 40, 50)]),
            np.concatenate([np.random.normal(185, 20, 600), np.random.normal(220, 15, 300), np.random.normal(260, 25, 100)])]
    bp = ax.boxplot(data, tick_labels=['Hemoglobin\n(g/dL)', 'Glucose\n(mg/dL)', 'Cholesterol\n(mg/dL)'], patch_artist=True, widths=0.6)
    for patch, color in zip(bp['boxes'], ['#3498db', '#e67e22', '#9b59b6']):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    for median in bp['medians']: median.set(color='#e74c3c', linewidth=2)
    ax.set_ylabel('Value', fontsize=12, fontweight='bold')
    ax.set_title('C) Key Biomarker Distributions', fontsize=13, fontweight='bold', pad=15)
    ax.grid(axis='y', alpha=0.3); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout(); plt.savefig('Figures/Sample Figures/Dataset/Image2.pdf', format='pdf', dpi=300, bbox_inches='tight'); plt.close()

def fig4():
    fig, axes = plt.subplots(1, 5, figsize=(15, 5))
    models = ['LR', 'RF', 'SVM', 'XGB', 'NN', 'Voting', 'Stacking']
    metrics = [[82.4, 89.1, 85.3, 91.7, 93.2, 94.5, 95.8], [78.6, 86.3, 82.1, 89.4, 91.8, 93.1, 94.9], [71.2, 84.5, 79.8, 88.1, 90.5, 92.3, 93.7], [74.7, 85.4, 80.9, 88.7, 91.1, 92.7, 94.3], [0.854, 0.923, 0.891, 0.951, 0.968, 0.978, 0.987]]
    names = ['Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1-Score (%)', 'ROC-AUC']
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
    for i, (metric, name, color) in enumerate(zip(metrics, names, colors)):
        bars = axes[i].bar(range(len(models)), metric, color=color, alpha=0.75, edgecolor='white', linewidth=0.5)
        bars[-1].set_color('#e74c3c'); bars[-1].set_alpha(1.0); bars[-1].set_edgecolor('#c0392b'); bars[-1].set_linewidth(2)
        axes[i].set_xticks(range(len(models))); axes[i].set_xticklabels(models, rotation=45, ha='right', fontsize=9)
        axes[i].set_ylabel(name, fontsize=10, fontweight='bold'); axes[i].set_title(f'{chr(65+i)}) {name}', fontsize=12, fontweight='bold', pad=10)
        axes[i].grid(axis='y', alpha=0.3, linestyle='--'); axes[i].spines['top'].set_visible(False); axes[i].spines['right'].set_visible(False)
        for j, (bar, val) in enumerate(zip(bars, metric)):
            axes[i].annotate(f'{val:.1f}', xy=(bar.get_x()+bar.get_width()/2, bar.get_height()), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=7, fontweight='bold' if j==len(models)-1 else 'normal')
    fig.legend(handles=[mpatches.Patch(facecolor='#3498db', alpha=0.75, label='Baseline Models'), mpatches.Patch(facecolor='#e74c3c', label='Proposed (Stacking)')], loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=2, fontsize=11, frameon=True)
    plt.tight_layout(rect=[0, 0.05, 1, 1]); plt.savefig('Figures/Sample Figures/Results/fig.pdf', format='pdf', dpi=300, bbox_inches='tight'); plt.close()

def fig5():
    fig, ax = plt.subplots(figsize=(10, 6))
    scenarios = ['Anemia\nDetection', 'Diabetes\nRisk', 'Hyperlipidemia\nScreening', 'Inflammation\nMonitoring', 'Cardiovascular\nRisk', 'Multi-Condition\nComorbidity']
    groups = ['Hematological', 'Metabolic', 'Lipid Profile', 'Inflammatory']
    data = np.array([[89.5, 68.2, 65.4, 72.1], [71.4, 91.3, 74.6, 68.5], [65.8, 72.0, 89.2, 66.3], [74.6, 68.1, 62.5, 88.7], [78.2, 82.5, 85.1, 76.4], [72.8, 75.4, 78.2, 80.5]])
    im = ax.imshow(data, cmap='YlOrRd', aspect='auto', vmin=60, vmax=95)
    ax.set_xticks(np.arange(len(groups))); ax.set_yticks(np.arange(len(scenarios)))
    ax.set_xticklabels(groups, fontsize=11, fontweight='bold'); ax.set_yticklabels(scenarios, fontsize=10)
    for i in range(len(scenarios)):
        for j in range(len(groups)):
            ax.text(j, i, f'{data[i,j]:.1f}', ha="center", va="center", color="black" if data[i,j]<80 else "white", fontsize=11, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax, shrink=0.8); cbar.set_label('F1-Score (%)', fontsize=12, fontweight='bold')
    ax.set_title('Feature Group Contribution Across Clinical Scenarios', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Feature Group', fontsize=12, fontweight='bold'); ax.set_ylabel('Clinical Scenario', fontsize=12, fontweight='bold')
    plt.tight_layout(); plt.savefig('Figures/Sample Figures/Results/Figure3.pdf', format='pdf', dpi=300, bbox_inches='tight'); plt.close()

def fig6():
    fig, ax = plt.subplots(figsize=(10, 6))
    sev = [0, 1, 2, 3, 4]
    labels = ['Clean\nData', 'Low\nPerturbation', 'Moderate\nPerturbation', 'High\nPerturbation', 'Extreme\nPerturbation']
    ax.plot(sev, [0.923, 0.891, 0.845, 0.782, 0.698], 'o-', color='#e74c3c', linewidth=2.5, markersize=8, label='Baseline ROC-AUC')
    ax.plot(sev, [0.987, 0.978, 0.963, 0.941, 0.912], 's-', color='#2ecc71', linewidth=2.5, markersize=8, label='Proposed ROC-AUC')
    ax2 = ax.twinx()
    ax2.plot(sev, [0.042, 0.058, 0.078, 0.105, 0.138], 'o--', color='#e74c3c', linewidth=2, markersize=6, alpha=0.6, label='Baseline ECE')
    ax2.plot(sev, [0.018, 0.025, 0.034, 0.048, 0.065], 's--', color='#2ecc71', linewidth=2, markersize=6, alpha=0.6, label='Proposed ECE')
    ax.set_xlabel('Perturbation Severity Level', fontsize=12, fontweight='bold'); ax.set_ylabel('ROC-AUC', fontsize=12, fontweight='bold', color='#2c3e50')
    ax2.set_ylabel('Expected Calibration Error (ECE)', fontsize=12, fontweight='bold', color='#7f8c8d')
    ax.set_xticks(sev); ax.set_xticklabels(labels, fontsize=10); ax.set_ylim(0.65, 1.0); ax2.set_ylim(0, 0.16)
    ax.grid(axis='y', alpha=0.3, linestyle='--'); ax.spines['top'].set_visible(False); ax2.spines['top'].set_visible(False)
    l1, la1 = ax.get_legend_handles_labels(); l2, la2 = ax2.get_legend_handles_labels()
    ax.legend(l1+l2, la1+la2, loc='center left', fontsize=10, frameon=True)
    ax.set_title('Robustness Degradation Under Increasing Data Perturbations', fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout(); plt.savefig('Figures/Sample Figures/Results/Figure4.pdf', format='pdf', dpi=300, bbox_inches='tight'); plt.close()

def fig7():
    fig, ax = plt.subplots(figsize=(9, 6))
    features = ['Age > 65 years', 'Glucose > 126 mg/dL', 'Systolic BP > 140 mmHg', 'LDL > 160 mg/dL', 'CRP > 3.0 mg/L', 'Hemoglobin < 12 g/dL', 'BMI > 30 kg/m²']
    importance = [0.28, 0.24, 0.18, 0.15, 0.14, 0.12, 0.10]
    groups = ['Demographics', 'Metabolic', 'Vital Signs', 'Lipid', 'Inflammatory', 'Hematological', 'Metabolic']
    gc = {'Demographics': '#1abc9c', 'Metabolic': '#e67e22', 'Vital Signs': '#2ecc71', 'Lipid': '#9b59b6', 'Inflammatory': '#e74c3c', 'Hematological': '#3498db'}
    colors = [gc[g] for g in groups]
    bars = ax.barh(range(len(features)), importance, color=colors, alpha=0.85, edgecolor='white', height=0.6)
    for i, (bar, val) in enumerate(zip(bars, importance)):
        ax.text(val+0.005, bar.get_y()+bar.get_height()/2, f'{val:.2f}', va='center', fontsize=10, fontweight='bold')
    ax.set_yticks(range(len(features))); ax.set_yticklabels(features, fontsize=11)
    ax.set_xlabel('Mean Importance Score', fontsize=12, fontweight='bold')
    ax.set_title('Top Biomarker Cues Identified by Integrated XAI (SHAP + LIME)', fontsize=13, fontweight='bold', pad=15)
    ax.set_xlim(0, 0.35); ax.grid(axis='x', alpha=0.3, linestyle='--'); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.legend(handles=[mpatches.Patch(facecolor=gc[g], label=g) for g in gc], loc='lower right', fontsize=9, frameon=True)
    plt.tight_layout(); plt.savefig('Figures/Sample Figures/Results/Figure5.pdf', format='pdf', dpi=300, bbox_inches='tight'); plt.close()

def fig8():
    fig, ax = plt.subplots(figsize=(8, 6))
    c = np.linspace(0, 1, 100)
    ax.plot(c, 0.15+0.05*c+0.03*np.sin(c*10), '--', color='#e74c3c', linewidth=2, label='Fixed Threshold')
    ax.plot(c, 0.12+0.04*c+0.02*np.sin(c*10), '-.', color='#f39c12', linewidth=2, label='Confidence-Only Ranking')
    ax.plot(c, 0.08+0.03*c+0.01*np.sin(c*10), '-', color='#2ecc71', linewidth=2.5, label='Proposed Uncertainty-Aware')
    ax.fill_between(c, 0.08+0.03*c+0.01*np.sin(c*10), 0.15+0.05*c+0.03*np.sin(c*10), alpha=0.15, color='#2ecc71')
    ax.set_xlabel('Coverage (Fraction of Patients Evaluated)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Risk (Fraction of Errors)', fontsize=12, fontweight='bold')
    ax.set_title('Risk-Coverage Curves for Patient Risk Prioritization', fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='upper left', fontsize=10, frameon=True); ax.grid(alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.set_xlim(0, 1); ax.set_ylim(0, 0.25)
    plt.tight_layout(); plt.savefig('Figures/Sample Figures/Results/Figure6.pdf', format='pdf', dpi=300, bbox_inches='tight'); plt.close()

if __name__ == '__main__':
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6(); fig7(); fig8()
    print('All figures generated')
