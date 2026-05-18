"""
Full Dataset Visualization from 1.02M Samples
Generates comprehensive visualizations showing all MMS data patterns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

print("=" * 80)
print("FULL DATASET VISUALIZATION (1,020,082 samples)")
print("=" * 80)

# Load full dataset
print("\n[1/5] Loading full dataset...")
csv_file = "labeled_sunside_data.csv"
df = pd.read_csv(csv_file)
print(f"✓ Loaded {len(df):,} samples")

# Extract features
print("\n[2/5] Extracting features from ALL samples...")
features = df[['norm_Btot', 'ratio_max_width', 'ratio_high_low']].copy()
labels = df['named_label'].copy()

# Remove NaN
mask = ~(features.isna().any(axis=1) | labels.isna())
features = features[mask]
labels = labels[mask]
print(f"✓ Valid samples: {len(features):,}")

# Map region names for consistency
region_map = {
    'SW': 'Solar Wind',
    'MSH': 'Magnetosheath',
    'MSP': 'Magnetosphere',
    'Foreshock': 'Ion Foreshock',
    'Unknown': 'Unknown'
}
labels_clean = labels.map(region_map).fillna(labels)

# Standardize
print("\n[3/5] Standardizing features...")
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
print(f"✓ Standardized {features_scaled.shape}")

print("\n[4/5] Creating comprehensive visualizations...")

# Create large figure with multiple subplots
fig = plt.figure(figsize=(20, 16))
fig.suptitle('MMS DAYSIDE DATA - FULL 1.02M SAMPLE ANALYSIS', 
             fontsize=20, fontweight='bold', y=0.995)

# Define colors for regions
region_colors = {
    'Solar Wind': '#1f77b4',
    'Magnetosheath': '#ff7f0e',
    'Magnetosphere': '#2ca02c',
    'Ion Foreshock': '#d62728',
    'Unknown': '#999999'
}

# ===== SUBPLOT 1: norm_Btot Distribution =====
ax1 = plt.subplot(3, 3, 1)
for region in labels_clean.unique():
    if pd.isna(region):
        continue
    mask = labels_clean == region
    data = features.loc[mask, 'norm_Btot']
    ax1.hist(data, bins=100, alpha=0.6, label=region, 
             color=region_colors.get(region, 'gray'),
             density=True)
ax1.set_xlabel('norm_Btot (B-field)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Density', fontsize=11, fontweight='bold')
ax1.set_title('B-Field Magnitude Distribution\n(All 1.02M samples)', fontsize=12, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)

# ===== SUBPLOT 2: ratio_max_width Distribution =====
ax2 = plt.subplot(3, 3, 2)
for region in labels_clean.unique():
    if pd.isna(region):
        continue
    mask = labels_clean == region
    data = features.loc[mask, 'ratio_max_width']
    ax2.hist(data, bins=100, alpha=0.6, label=region,
             color=region_colors.get(region, 'gray'),
             density=True)
ax2.set_xlabel('ratio_max_width (Spectral Width)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Density', fontsize=11, fontweight='bold')
ax2.set_title('Ion Spectral Width Distribution\n(All 1.02M samples)', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)

# ===== SUBPLOT 3: ratio_high_low Distribution =====
ax3 = plt.subplot(3, 3, 3)
for region in labels_clean.unique():
    if pd.isna(region):
        continue
    mask = labels_clean == region
    data = features.loc[mask, 'ratio_high_low']
    ax3.hist(data, bins=100, alpha=0.6, label=region,
             color=region_colors.get(region, 'gray'),
             density=True)
ax3.set_xlabel('ratio_high_low (Energy Ratio)', fontsize=11, fontweight='bold')
ax3.set_ylabel('Density', fontsize=11, fontweight='bold')
ax3.set_title('Energy Ratio Distribution\n(All 1.02M samples)', fontsize=12, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(alpha=0.3)

# ===== SUBPLOT 4: 2D Scatter - norm_Btot vs ratio_max_width =====
ax4 = plt.subplot(3, 3, 4)
for region in labels_clean.unique():
    if pd.isna(region):
        continue
    mask = labels_clean == region
    ax4.scatter(features.loc[mask, 'norm_Btot'], 
               features.loc[mask, 'ratio_max_width'],
               alpha=0.3, s=1, label=region,
               color=region_colors.get(region, 'gray'))
ax4.set_xlabel('norm_Btot', fontsize=11, fontweight='bold')
ax4.set_ylabel('ratio_max_width', fontsize=11, fontweight='bold')
ax4.set_title('B-Field vs Spectral Width\n(All 1.02M samples)', fontsize=12, fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(alpha=0.3)

# ===== SUBPLOT 5: 2D Scatter - norm_Btot vs ratio_high_low =====
ax5 = plt.subplot(3, 3, 5)
for region in labels_clean.unique():
    if pd.isna(region):
        continue
    mask = labels_clean == region
    ax5.scatter(features.loc[mask, 'norm_Btot'],
               features.loc[mask, 'ratio_high_low'],
               alpha=0.3, s=1, label=region,
               color=region_colors.get(region, 'gray'))
ax5.set_xlabel('norm_Btot', fontsize=11, fontweight='bold')
ax5.set_ylabel('ratio_high_low', fontsize=11, fontweight='bold')
ax5.set_title('B-Field vs Energy Ratio\n(All 1.02M samples)', fontsize=12, fontweight='bold')
ax5.legend(fontsize=9)
ax5.grid(alpha=0.3)

# ===== SUBPLOT 6: 2D Scatter - ratio_max_width vs ratio_high_low =====
ax6 = plt.subplot(3, 3, 6)
for region in labels_clean.unique():
    if pd.isna(region):
        continue
    mask = labels_clean == region
    ax6.scatter(features.loc[mask, 'ratio_max_width'],
               features.loc[mask, 'ratio_high_low'],
               alpha=0.3, s=1, label=region,
               color=region_colors.get(region, 'gray'))
ax6.set_xlabel('ratio_max_width', fontsize=11, fontweight='bold')
ax6.set_ylabel('ratio_high_low', fontsize=11, fontweight='bold')
ax6.set_title('Spectral Width vs Energy Ratio\n(All 1.02M samples)', fontsize=12, fontweight='bold')
ax6.legend(fontsize=9)
ax6.grid(alpha=0.3)

# ===== SUBPLOT 7: Region Distribution (Pie Chart) =====
ax7 = plt.subplot(3, 3, 7)
region_counts = labels_clean.value_counts()
colors_pie = [region_colors.get(r, 'gray') for r in region_counts.index]
explode = [0.05 if r == 'Solar Wind' else 0.02 for r in region_counts.index]
ax7.pie(region_counts.values, labels=region_counts.index, autopct='%1.1f%%',
        colors=colors_pie, explode=explode, startangle=90, textprops={'fontsize': 10})
ax7.set_title('Region Distribution\n(1.02M samples)', fontsize=12, fontweight='bold')

# ===== SUBPLOT 8: Region Distribution (Bar Chart) =====
ax8 = plt.subplot(3, 3, 8)
region_counts = labels_clean.value_counts()
bars = ax8.bar(range(len(region_counts)), region_counts.values, 
               color=[region_colors.get(r, 'gray') for r in region_counts.index])
ax8.set_xticks(range(len(region_counts)))
ax8.set_xticklabels(region_counts.index, rotation=45, ha='right', fontsize=10)
ax8.set_ylabel('Number of Samples', fontsize=11, fontweight='bold')
ax8.set_title('Sample Count by Region\n(All 1.02M)', fontsize=12, fontweight='bold')
ax8.grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax8.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height):,}',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

# ===== SUBPLOT 9: Summary Statistics Table =====
ax9 = plt.subplot(3, 3, 9)
ax9.axis('off')

# Create summary statistics
summary_data = []
summary_data.append(['DATASET STATISTICS', '', '', ''])
summary_data.append(['', '', '', ''])
summary_data.append(['Total Samples', f'{len(features):,}', '', ''])
summary_data.append(['Valid Samples', f'{len(features):,}', '', ''])
summary_data.append(['', '', '', ''])
summary_data.append(['REGION COUNTS', '', '', ''])
for region in region_counts.index:
    count = region_counts[region]
    pct = 100 * count / len(features)
    summary_data.append([region, f'{count:,}', f'{pct:.1f}%', ''])

summary_data.append(['', '', '', ''])
summary_data.append(['FEATURE STATISTICS', 'Mean', 'Std', 'Range'])
for feat in ['norm_Btot', 'ratio_max_width', 'ratio_high_low']:
    data = features[feat]
    summary_data.append([feat, f'{data.mean():.3f}', f'{data.std():.3f}', 
                        f'{data.min():.3f}-{data.max():.3f}'])

# Create table
table = ax9.table(cellText=summary_data, cellLoc='left', loc='center',
                  colWidths=[0.35, 0.25, 0.2, 0.2])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)

# Style header rows
for i in range(len(summary_data)):
    for j in range(4):
        cell = table[(i, j)]
        if i in [0, 5, 10]:  # Header rows
            cell.set_facecolor('#4472C4')
            cell.set_text_props(weight='bold', color='white')
        elif i % 2 == 0:
            cell.set_facecolor('#E7E6E6')
        else:
            cell.set_facecolor('#F2F2F2')

plt.tight_layout()
print("✓ Creating comprehensive visualization...")

# Save figure
output_file = "full_dataset_analysis.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\n[5/5] ✓ Saved: {output_file}")

# Print summary statistics
print("\n" + "=" * 80)
print("SUMMARY STATISTICS (Full 1.02M Dataset)")
print("=" * 80)
print(f"\nTotal Samples: {len(features):,}")
print(f"Valid Samples: {len(features):,}")
print(f"\nRegion Distribution:")
for region in region_counts.index:
    count = region_counts[region]
    pct = 100 * count / len(features)
    print(f"  {region:20s}: {count:>10,} ({pct:>5.1f}%)")

print(f"\nFeature Statistics:")
print(f"  {' ':20s} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
for feat in features.columns:
    data = features[feat]
    print(f"  {feat:20s} {data.mean():>12.6f} {data.std():>12.6f} "
          f"{data.min():>12.6f} {data.max():>12.6f}")

print(f"\nStatistics by Region:")
for region in labels_clean.unique():
    if pd.isna(region):
        continue
    mask = labels_clean == region
    region_features = features[mask]
    print(f"\n  {region}:")
    for feat in features.columns:
        data = region_features[feat]
        print(f"    {feat:18s} - Mean: {data.mean():.4f}, Std: {data.std():.4f}, "
              f"Range: [{data.min():.4f}, {data.max():.4f}]")

print("\n" + "=" * 80)
print("✓ FULL DATASET VISUALIZATION COMPLETE")
print("=" * 80)
plt.show()
