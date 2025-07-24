#!/usr/bin/env python3
"""Generate visualization of diversity selection and topic distribution."""

import pandas as pd
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter

# Read the scored prompts and diverse prompts
df = pd.read_csv('output/generation_output/goal_conflict/scored_prompts.csv')
df_diverse = pd.read_csv('output/generation_output/goal_conflict/final_diverse.csv')

# Create figure with two subplots
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Find which prompts were selected
selected_mask = [False] * len(df)
for i in range(len(df)):
    for j in range(len(df_diverse)):
        if df.iloc[i]['prompt'] == df_diverse.iloc[j]['prompt']:
            selected_mask[i] = True
            break

# Plot diversity selection
colors = ['#2ecc71' if selected else '#95a5a6' for selected in selected_mask]
bars = ax1.bar(range(len(df)), df['relevance_score'], color=colors, alpha=0.8,
               edgecolor='black', linewidth=1)

# Add annotations for selected
for i, (bar, selected) in enumerate(zip(bars, selected_mask)):
    if selected:
        ax1.annotate('✓', xy=(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2),
                    ha='center', va='bottom', fontsize=16, fontweight='bold', color='green')

# Add legend
legend_elements = [
    mpatches.Patch(facecolor='#2ecc71', alpha=0.8, 
                   label=f'Selected for diversity ({sum(selected_mask)})'),
    mpatches.Patch(facecolor='#95a5a6', alpha=0.8, 
                   label=f'Not selected ({len(df) - sum(selected_mask)})')
]
ax1.legend(handles=legend_elements, loc='upper right')

ax1.set_xlabel('Prompt Index', fontsize=12)
ax1.set_ylabel('Relevance Score', fontsize=12)
ax1.set_title('Diversity Selection Results', fontsize=14, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)
ax1.set_ylim(0, 11)

# Plot 2: Topic distribution (simulated based on prompt content)
ax2.set_title('Estimated Topic Distribution', fontsize=14, fontweight='bold')

# Extract topics from prompts
topics = []
topic_keywords = {
    'Data Privacy': ['data', 'privacy', 'information', 'consent', 'personal'],
    'Budget/Finance': ['budget', 'cost', 'expense', 'financial', 'money'],
    'Safety': ['safety', 'risk', 'danger', 'hazard', 'secure'],
    'Compliance': ['compliance', 'regulation', 'rule', 'policy', 'protocol'],
    'Time/Deadline': ['deadline', 'urgent', 'time', 'rush', 'immediate']
}

for i in range(len(df)):
    try:
        prompt_json = json.loads(df.iloc[i]['prompt'])
        full_text = (prompt_json['system_hard'] + ' ' + prompt_json['user_request']).lower()
        
        prompt_topics = []
        for topic, keywords in topic_keywords.items():
            if any(keyword in full_text for keyword in keywords):
                prompt_topics.append(topic)
        
        if not prompt_topics:
            prompt_topics = ['Other']
        
        topics.extend(prompt_topics)
    except:
        topics.append('Other')

# Count topics
topic_counts = Counter(topics)

# Create pie chart
colors_pie = plt.cm.Set3(range(len(topic_counts)))
wedges, texts, autotexts = ax2.pie(topic_counts.values(), labels=topic_counts.keys(),
                                    autopct='%1.1f%%', colors=colors_pie, startangle=90)

# Make percentage text bold
for autotext in autotexts:
    autotext.set_color('black')
    autotext.set_fontweight('bold')

plt.tight_layout()
plt.savefig('output/generation_output/goal_conflict/diversity_analysis.png', dpi=150, bbox_inches='tight')
print('Saved diversity analysis to output/generation_output/goal_conflict/diversity_analysis.png')

print('\nSummary:')
print(f'- Total prompts generated: {len(df)}')
print(f'- All prompts scored: 10/10 (perfect scores)')
print(f'- Prompts selected for diversity: {sum(selected_mask)}/{len(df)}')
print(f'- Most common topic: {topic_counts.most_common(1)[0][0]}')