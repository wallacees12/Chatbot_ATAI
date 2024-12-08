import pandas as pd
import pprint

def compute_inter_rater_agreement(correct_count, incorrect_count):
    # Example heuristic:
    # Agreement = max(correct_count, incorrect_count) / (correct_count + incorrect_count)
    # Adjust this to a more complex metric if desired (e.g., Fleiss' Kappa)
    total = correct_count + incorrect_count
    if total == 0:
        return 0.0
    return max(correct_count, incorrect_count) / total

def process_crowd_data(crowd_data_path):
    # Read the CSV data
    df = pd.read_csv(crowd_data_path, sep='\t')  # adjust delimiter if needed
    
    # Assume each triple is identified by (Input1ID, Input2ID, Input3ID)
    grouping_cols = ['Input1ID', 'Input2ID', 'Input3ID']
    
    # Aggregate results per triple
    agg = df.groupby(grouping_cols).apply(lambda g: pd.Series({
        'correct_count': (g['AnswerLabel'] == 'CORRECT').sum(),
        'incorrect_count': (g['AnswerLabel'] == 'INCORRECT').sum()
    })).reset_index()
    
    # Compute inter-rater agreement and prepare distribution text
    agg['inter_rater_agreement'] = agg.apply(lambda row: compute_inter_rater_agreement(row['correct_count'], row['incorrect_count']), axis=1)
    agg['answer_distribution'] = agg.apply(lambda row: f"{row['correct_count']} support votes, {row['incorrect_count']} reject votes", axis=1)
    
    # Convert to a dictionary keyed by triple for easy lookup
    results_dict = {}
    for _, row in agg.iterrows():
        triple_key = (row['Input1ID'], row['Input2ID'], row['Input3ID'])
        results_dict[triple_key] = {
            'inter_rater_agreement': row['inter_rater_agreement'],
            'answer_distribution': row['answer_distribution'],
            'correct_count': row['correct_count'],
            'incorrect_count': row['incorrect_count']
        }
    
    return results_dict

# Example usage:
# Suppose the triple corresponds to a question you answered:
# "What is the box office of The Princess and the Frog?"
# and the triple for that question is (wd:Q11621, wdt:P2142, 792910554)

crowd_results = process_crowd_data('crowd_data.tsv')  # path to your TSV file

# Retrieve the aggregated crowd info for a given triple
triple_key = ('wd:Q11621', "wdt:P2142", "792910554")
print(f"{triple_key}")
if triple_key in crowd_results:
    irr = crowd_results[triple_key]['inter_rater_agreement']
    dist = crowd_results[triple_key]['answer_distribution']
    # Integrate this into your final answer:
    # For example, if your answer was "The box office is 267000000."
    answer = "The box office of The Princess and the Frog is 267000000."
    # Append crowd data
    answer += f"\n[Crowd, inter-rater agreement {irr:.3f}, The answer distribution for this specific task was {dist}]"
    print(answer)