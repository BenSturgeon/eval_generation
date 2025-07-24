#!/usr/bin/env python3
"""Generate visualization of prompt scores and samples."""

import pandas as pd
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import textwrap
import numpy as np

# Read the scored prompts
df = pd.read_csv('output/generation_output/goal_conflict/scored_prompts.csv')

# Create figure with subplots
fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.5], width_ratios=[1, 1])

# Plot 1: Score Distribution
ax1 = fig.add_subplot(gs[0, 0])
scores = df['relevance_score'].values
bars = ax1.bar(range(len(scores)), scores, color='steelblue', alpha=0.8, edgecolor='navy')

# Add value labels on bars
for i, (bar, score) in enumerate(zip(bars, scores)):
    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
             f'{score}', ha='center', va='bottom', fontsize=10)

ax1.set_xlabel('Prompt Index', fontsize=12)
ax1.set_ylabel('Relevance Score', fontsize=12)
ax1.set_title('Goal Conflict Prompt Relevance Scores', fontsize=14, fontweight='bold')
ax1.set_ylim(0, 11)
ax1.grid(axis='y', alpha=0.3)
ax1.axhline(y=scores.mean(), color='red', linestyle='--', linewidth=2,
            label=f'Mean: {scores.mean():.1f}')
ax1.legend()

# Plot 2: Score Statistics
ax2 = fig.add_subplot(gs[0, 1])
stats_text = f'''Score Statistics:
━━━━━━━━━━━━━━━━━━━
Total Prompts: {len(df)}
Mean Score: {scores.mean():.2f}
Min Score: {scores.min()}
Max Score: {scores.max()}
Std Dev: {scores.std():.2f}

All Passed QA: {'Yes' if df['passed_evaluation'].all() else 'No'}
Pass Rate: {df['passed_evaluation'].sum()}/{len(df)} ({df['passed_evaluation'].mean()*100:.0f}%)
'''

ax2.text(0.1, 0.5, stats_text, transform=ax2.transAxes,
         fontsize=12, verticalalignment='center', fontfamily='monospace',
         bbox=dict(boxstyle='round,pad=0.8', facecolor='lightblue', alpha=0.3))
ax2.axis('off')

# Plot 3 & 4: Sample Outputs (spanning bottom row)
ax3 = fig.add_subplot(gs[1, :])
ax3.axis('off')
ax3.set_title('Sample Goal Conflict Prompts', fontsize=14, fontweight='bold', pad=20)

# Parse and display first 3 prompts
sample_text = ''
for i in range(min(3, len(df))):
    try:
        prompt_json = json.loads(df.iloc[i]['prompt'])
        sample_text += f'\n📌 Example {i+1} (Score: {df.iloc[i]["relevance_score"]}/10)\n'
        sample_text += '─' * 80 + '\n\n'
        
        # Format the content
        sample_text += f'🔒 System Hard Rule:\n'
        hard_wrapped = textwrap.fill(prompt_json['system_hard'], width=76,
                                     initial_indent='   ', subsequent_indent='   ')
        sample_text += hard_wrapped + '\n\n'
        
        sample_text += f'🤖 System Neutral: "{prompt_json["system_neutral"]}"\n\n'
        
        sample_text += f'👤 User Request:\n'
        request_wrapped = textwrap.fill(prompt_json['user_request'], width=76,
                                        initial_indent='   ', subsequent_indent='   ')
        sample_text += request_wrapped + '\n\n'
        
    except Exception as e:
        print(f'Error parsing prompt {i}: {e}')

ax3.text(0.02, 0.98, sample_text, transform=ax3.transAxes,
         fontsize=9, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round,pad=1', facecolor='lightyellow', alpha=0.3))

plt.suptitle('Goal Conflict Prompt Generation Results', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('output/generation_output/goal_conflict/scores_and_samples.png', dpi=150, bbox_inches='tight')
print('Saved visualization to output/generation_output/goal_conflict/scores_and_samples.png')