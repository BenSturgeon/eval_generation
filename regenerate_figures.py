#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Read the latest scored prompts
df = pd.read_csv('output/generation_output/goal_conflict/scored_prompts.csv')
print(f"Loaded {len(df)} prompts from scored_prompts.csv")

# Figure 1: Score distribution and samples
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Score distribution
score_counts = df['relevance_score'].value_counts().sort_index()
ax1.bar(score_counts.index, score_counts.values, color='steelblue', alpha=0.8)
ax1.set_xlabel('Relevance Score')
ax1.set_ylabel('Count')
ax1.set_title(f'Distribution of Relevance Scores (n={len(df)})')
ax1.grid(axis='y', alpha=0.3)

# Sample prompts from different score ranges
ax2.axis('off')
ax2.set_title('Sample Prompts by Score Range', fontsize=14, pad=20)

y_position = 0.95
samples_shown = 0
max_samples = 5

# Get samples from different score ranges
for score in sorted(df['relevance_score'].unique(), reverse=True):
    if samples_shown >= max_samples:
        break
    
    score_df = df[df['relevance_score'] == score]
    if len(score_df) > 0:
        sample = score_df.iloc[0]
        
        # Parse the prompt JSON
        import json
        try:
            prompt_data = json.loads(sample['prompt'])
            
            # Display score header
            ax2.text(0.02, y_position, f'Score: {score}/10', 
                    fontsize=12, weight='bold', transform=ax2.transAxes)
            y_position -= 0.04
            
            # Display system_hard (truncated)
            system_hard = prompt_data.get('system_hard', '')[:100] + '...'
            ax2.text(0.02, y_position, f'System: {system_hard}', 
                    fontsize=9, wrap=True, transform=ax2.transAxes)
            y_position -= 0.04
            
            # Display user_request (truncated)
            user_request = prompt_data.get('user_request', '')[:100] + '...'
            ax2.text(0.02, y_position, f'User: {user_request}', 
                    fontsize=9, wrap=True, transform=ax2.transAxes)
            y_position -= 0.08
            
            samples_shown += 1
        except:
            pass

plt.tight_layout()
plt.savefig('output/generation_output/goal_conflict/scores_and_samples.png', dpi=150, bbox_inches='tight')
plt.close()
print("Generated scores_and_samples.png")

# Figure 2: Diversity analysis (if diverse prompts exist)
if os.path.exists('output/generation_output/goal_conflict/final_diverse.csv'):
    diverse_df = pd.read_csv('output/generation_output/goal_conflict/final_diverse.csv')
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Create a simple visualization showing topic distribution
    topics = []
    for idx, row in diverse_df.iterrows():
        try:
            prompt_data = json.loads(row['prompt'])
            # Extract topic hints from system_hard
            system_hard = prompt_data.get('system_hard', '').lower()
            
            # Simple topic detection based on keywords
            if 'budget' in system_hard or 'cost' in system_hard or 'expense' in system_hard:
                topics.append('Budget/Finance')
            elif 'data' in system_hard or 'privacy' in system_hard or 'information' in system_hard:
                topics.append('Data/Privacy')
            elif 'energy' in system_hard or 'power' in system_hard or 'consumption' in system_hard:
                topics.append('Energy/Resources')
            elif 'time' in system_hard or 'deadline' in system_hard or 'schedule' in system_hard:
                topics.append('Time/Schedule')
            elif 'security' in system_hard or 'access' in system_hard or 'permission' in system_hard:
                topics.append('Security/Access')
            else:
                topics.append('Other')
        except:
            topics.append('Unknown')
    
    # Plot topic distribution
    topic_counts = pd.Series(topics).value_counts()
    colors = sns.color_palette("husl", len(topic_counts))
    
    ax.pie(topic_counts.values, labels=topic_counts.index, autopct='%1.0f%%', 
           colors=colors, startangle=90)
    ax.set_title(f'Topic Distribution in {len(diverse_df)} Diverse Prompts', fontsize=14, pad=20)
    
    # Add summary text
    summary_text = f"Total diverse prompts: {len(diverse_df)}\\n"
    summary_text += f"Average score: {diverse_df['relevance_score'].mean():.1f}\\n"
    summary_text += f"Score range: {diverse_df['relevance_score'].min()}-{diverse_df['relevance_score'].max()}"
    ax.text(1.3, 0.5, summary_text, transform=ax.transAxes, fontsize=11,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('output/generation_output/goal_conflict/diversity_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated diversity_analysis.png")
else:
    print("No diverse prompts file found, skipping diversity analysis")

print("\\nFigure regeneration complete!")