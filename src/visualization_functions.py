from io import StringIO

from plotly import graph_objects as go
import numpy as np
import pandas as pd
import umap.umap_ as umap

from src.utils import create_collapsible_html_list

PLOT_HEIGHTS = 500


def get_fig_html_as_string(fig: go.Figure):
    html_buffer = StringIO()
    fig.write_html(html_buffer, full_html=False)
    return html_buffer.getvalue()


def visualize_scores(df):
    df = df.sort_values('relevance_score', ascending=False)
    df['prompt'] = df['prompt'].apply(lambda x: str(x['text']) if isinstance(x, dict) else str(x)) # Jacy
    plot_data = {
        "Generated_prompts": {
            x['prompt']: [
                'System Prompt: ' + x['system_prompt'],
                'Generative Prompt: ' + x['generative_prompt'],
                {
                    f'Relevance: {x["relevance_score"]}': [
                        f'Relevance prompt: {x["relevance_prompt"]}',
                        f'Model Response: {x["model_response"]}',
                    ]
                }
            ]
            for _, x in df.iterrows()
        }
    }

    hist_fig = go.Figure(
        go.Histogram(x=df['relevance_score'], name='Relevance')
    )
    hist_fig.update_layout(
        title="Prompt Evaluation Scores Histogram",
        xaxis_title="Score",
        yaxis_title="Frequency",
        height=PLOT_HEIGHTS
    )

    range = np.arange(len(df))
    scatter_fig = go.Figure(
        [
            go.Scatter(
                x=range[df['passed_evaluation']],
                y=df['relevance_score'][df['passed_evaluation']],
                mode='markers',
                text=df['prompt'][df['passed_evaluation']],
                hovertemplate='<b>Score</b>: %{y}<br>' +
                            '<b>Prompt</b>: %{text}<br>' +
                            '<extra></extra>',
                name="In top n relevant prompts"
            ),
            go.Scatter(
                x=df.index[~df['passed_evaluation']],
                y=df['relevance_score'][~df['passed_evaluation']],
                mode='markers',
                text=df['prompt'][~df['passed_evaluation']],
                hovertemplate='<b>Score</b>: %{y}<br>' +
                            '<b>Prompt</b>: %{text}<br>' +
                            '<extra></extra>',
                name="Not in top n relevant prompts"
            ),
        ]
    )
    scatter_fig.update_layout(
        title="Prompt Evaluation Scores Scatter Plot",
        xaxis_title="Prompt Index",
        yaxis_title="Score (1-1000)",
        hovermode='closest',
        height=600,
        width=1200,
    )

    hist_fig_html = get_fig_html_as_string(hist_fig)
    scatter_fig_html = get_fig_html_as_string(scatter_fig)

    html_str = create_collapsible_html_list(plot_data) + hist_fig_html + scatter_fig_html

    return html_str


def visualize_diversity(df: pd.DataFrame, representative_samples: list, pca_features: np.ndarray, cluster: np.ndarray) -> str:

    umap_reducer = umap.UMAP(n_components=2, random_state=42)
    vis_dims = umap_reducer.fit(np.array(pca_features)).embedding_

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=vis_dims[:, 0], y=vis_dims[:, 1], mode='markers',
            text=df['prompt'], hoverinfo='text',
            marker=dict(color=cluster, colorscale='Viridis', size=5, opacity=0.7),
            name='Non Representative Prompts'
        )
    )
    fig.add_trace(
        go.Scatter(
            x=vis_dims[representative_samples, 0], y=vis_dims[representative_samples, 1], mode='markers',
            text=df['prompt'][representative_samples], hoverinfo='text',
            marker=dict(color='red', size=10, opacity=0.9),
            name='Representative Prompts'
        )
    )
    fig.update_layout(
        title="UMAP Visualization of Text Embeddings with k-Means Clustering",
        xaxis_title="Component 1", yaxis_title="Component 2", hovermode='closest', height=PLOT_HEIGHTS
    )

    fig_html = get_fig_html_as_string(fig)

    return fig_html


def create_representative_prompts_html(is_diverse_df: pd.DataFrame) -> str:
    is_diverse_df['prompt'] = is_diverse_df['prompt'].apply(lambda x: str(x['text']) if isinstance(x, dict) else str(x)) # Jacy
    plot_data = {
        "Representative Prompts ": {
            x['prompt']: [
                'system_prompt: ' + x['system_prompt'],
                'generative_prompt: ' + x['generative_prompt'],
                {
                    f'relevance_score: {x["relevance_score"]}': [
                        f'relevance_system_prompt: {x["relevance_system_prompt"]}',
                        f'relevance_prompt: {x["relevance_prompt"]}'
                        f'model_response: {x["model_response"]}',
                    ],
                },
            ]
            for _, x in is_diverse_df.iterrows()
        }
    }
    return create_collapsible_html_list(plot_data)


def create_subject_responses_html(is_diverse_df: pd.DataFrame, subject_model, best_possible_score) -> str:
    plot_data = {
        f"Subject responses ({subject_model})": [
            f"Best possible score: {best_possible_score}",
            [
                {
                    f"Score: {x['score']}": [
                        f"Prompt: {x['prompt']}",
                        f"Subject system prompt: {x['subject_system_prompt']}",
                        f"Evaluator system prompt: {x['evaluator_system_prompt']}",
                        f"Evaluator prompt: {x['evaluator_prompt']}"
                        f"Evaluator response: {x['evaluator_response']}"
                    ]
                }
                for _, x in is_diverse_df.sort_values('score', ascending=False).iterrows()
            ]
        ]
    }

    return create_collapsible_html_list(plot_data)


def visualize_subject_model_scores(df: pd.DataFrame, subject_models: list) -> str:
    fig = go.Figure()

    for subject_model in subject_models:
        subject_df = df[df['subject_model'] == subject_model]
        fig.add_trace(
            go.Histogram(x=subject_df['score'], name=subject_model)
        )

    fig.update_layout(
        title="Histogram of Model Scores",
        xaxis_title="Score",
        yaxis_title="Frequency",
        height=PLOT_HEIGHTS
    )

    html_str = StringIO()
    fig.write_html(html_str, full_html=False)
    print("v", html_str.getvalue()[:10])
    return html_str.getvalue()


def visualize_subject_model_responses(df: pd.DataFrame, subject_models: list, best_possible_score_for_problem) -> str:
    html_out = ""
    for subject_model in subject_models:
        subject_df = df[df['subject_model'] == subject_model]
        html_out += create_subject_responses_html(subject_df, subject_model, best_possible_score_for_problem)

    return html_out


def get_mean_model_scores(df: pd.DataFrame, subject_models: list, best_possible_score_for_problem) -> str:
    html_scores = ""
    for subject_model in subject_models:
        subject_df = df[df['subject_model'] == subject_model]
        html_scores += f"<h3>{subject_model} Mean Score: {subject_df['score'].mean() / best_possible_score_for_problem * 100:.2f}%</h3>"

    return html_scores


def generate_png_visualizations(df_scores: pd.DataFrame, df_diverse: pd.DataFrame = None, output_folder: str = None):
    """Generate PNG visualizations for scores and diversity analysis."""
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import json
    import textwrap
    from collections import Counter
    
    # Generate scores and samples visualization
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.5], width_ratios=[1, 1])
    
    # Plot 1: Score Distribution
    ax1 = fig.add_subplot(gs[0, 0])
    scores = df_scores['relevance_score'].values
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
Total Prompts: {len(df_scores)}
Mean Score: {scores.mean():.2f}
Min Score: {scores.min()}
Max Score: {scores.max()}
Std Dev: {scores.std():.2f}

All Passed QA: {'Yes' if df_scores['passed_evaluation'].all() else 'No'}
Pass Rate: {df_scores['passed_evaluation'].sum()}/{len(df_scores)} ({df_scores['passed_evaluation'].mean()*100:.0f}%)
'''
    
    ax2.text(0.1, 0.5, stats_text, transform=ax2.transAxes,
             fontsize=12, verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.8', facecolor='lightblue', alpha=0.3))
    ax2.axis('off')
    
    # Plot 3 & 4: Sample Outputs - Failed and Passed prompts
    # Left subplot - Failed prompts
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.axis('off')
    ax3.set_title('Failed Prompts (Score < threshold)', fontsize=12, fontweight='bold', pad=10)
    
    # Get failed prompts (those that didn't pass evaluation)
    failed_df = df_scores[~df_scores['passed_evaluation']].sort_values('relevance_score')
    
    # Parse and display first 3 failed prompts
    failed_text = ''
    for i in range(min(3, len(failed_df))):
        try:
            prompt_str = failed_df.iloc[i]['prompt']
            # Handle both dict and string formats
            if isinstance(prompt_str, dict):
                prompt_json = prompt_str
            else:
                prompt_json = json.loads(prompt_str)
            failed_text += f'\nExample {i+1} (Score: {failed_df.iloc[i]["relevance_score"]}/10)\n'
            failed_text += '─' * 38 + '\n\n'
            
            # Format the content
            failed_text += f'System Hard Rule:\n'
            hard_wrapped = textwrap.fill(prompt_json['system_hard'], width=36,
                                         initial_indent='  ', subsequent_indent='  ')
            failed_text += hard_wrapped + '\n\n'
            
            failed_text += f'System Neutral: "{prompt_json["system_neutral"]}"\n\n'
            
            failed_text += f'User Request:\n'
            request_wrapped = textwrap.fill(prompt_json['user_request'], width=36,
                                            initial_indent='  ', subsequent_indent='  ')
            failed_text += request_wrapped + '\n\n'
            
        except Exception as e:
            print(f'Error parsing failed prompt {i}: {e}')
    
    if len(failed_df) == 0:
        failed_text = '\nNo failed prompts - all prompts passed evaluation!'
    
    ax3.text(0.02, 0.98, failed_text, transform=ax3.transAxes,
             fontsize=8, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.8', facecolor='lightcoral', alpha=0.3))
    
    # Right subplot - Passed prompts
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    ax4.set_title('Passed Prompts (Score ≥ threshold)', fontsize=12, fontweight='bold', pad=10)
    
    # Get passed prompts (those that passed evaluation)
    passed_df = df_scores[df_scores['passed_evaluation']].sort_values('relevance_score', ascending=False)
    
    # Parse and display first 3 passed prompts
    passed_text = ''
    for i in range(min(3, len(passed_df))):
        try:
            prompt_str = passed_df.iloc[i]['prompt']
            # Handle both dict and string formats
            if isinstance(prompt_str, dict):
                prompt_json = prompt_str
            else:
                prompt_json = json.loads(prompt_str)
            passed_text += f'\nExample {i+1} (Score: {passed_df.iloc[i]["relevance_score"]}/10)\n'
            passed_text += '─' * 38 + '\n\n'
            
            # Format the content
            passed_text += f'System Hard Rule:\n'
            hard_wrapped = textwrap.fill(prompt_json['system_hard'], width=36,
                                         initial_indent='  ', subsequent_indent='  ')
            passed_text += hard_wrapped + '\n\n'
            
            passed_text += f'System Neutral: "{prompt_json["system_neutral"]}"\n\n'
            
            passed_text += f'User Request:\n'
            request_wrapped = textwrap.fill(prompt_json['user_request'], width=36,
                                            initial_indent='  ', subsequent_indent='  ')
            passed_text += request_wrapped + '\n\n'
            
        except Exception as e:
            print(f'Error parsing passed prompt {i}: {e}')
    
    if len(passed_df) == 0:
        passed_text = '\nNo passed prompts - all prompts failed evaluation!'
    
    ax4.text(0.02, 0.98, passed_text, transform=ax4.transAxes,
             fontsize=8, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.8', facecolor='lightgreen', alpha=0.3))
    
    plt.suptitle('Goal Conflict Prompt Generation Results', fontsize=16, fontweight='bold')
    plt.tight_layout()
    scores_path = f'{output_folder}/scores_and_samples.png'
    plt.savefig(scores_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Generated PNG visualization: {scores_path}')
    
    # Generate diversity visualization if diverse prompts provided
    if df_diverse is not None and len(df_diverse) > 0:
        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Find which prompts were selected
        selected_mask = [False] * len(df_scores)
        for i in range(len(df_scores)):
            for j in range(len(df_diverse)):
                if df_scores.iloc[i]['prompt'] == df_diverse.iloc[j]['prompt']:
                    selected_mask[i] = True
                    break
        
        # Plot diversity selection
        colors = ['#2ecc71' if selected else '#95a5a6' for selected in selected_mask]
        bars = ax1.bar(range(len(df_scores)), df_scores['relevance_score'], color=colors, alpha=0.8,
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
                           label=f'Not selected ({len(df_scores) - sum(selected_mask)})')
        ]
        ax1.legend(handles=legend_elements, loc='upper right')
        
        ax1.set_xlabel('Prompt Index', fontsize=12)
        ax1.set_ylabel('Relevance Score', fontsize=12)
        ax1.set_title('Diversity Selection Results', fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_ylim(0, 11)
        
        # Plot 2: Topic distribution
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
        
        for i in range(len(df_scores)):
            try:
                prompt_str = df_scores.iloc[i]['prompt']
                if isinstance(prompt_str, dict):
                    prompt_json = prompt_str
                else:
                    prompt_json = json.loads(prompt_str)
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
        diversity_path = f'{output_folder}/diversity_analysis.png'
        plt.savefig(diversity_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'Generated PNG visualization: {diversity_path}')
