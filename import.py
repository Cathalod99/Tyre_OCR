def anova_test_exam_scores(scores_data):  
    if not scores_data or len(scores_data) < 2:  
        return {'F-statistic': None, 'P-value': None}  

    num_groups = 0  
    total_samples = 0  
    group_means = []  
    group_counts = []  
    grand_mean_list = []  

    # Collect data  
    for school, data in scores_data.items():  
        if not data:  
            continue  

        num_groups += 1  
        sum_data = sum(data)  
        count_data = len(data)  
        mean_data = sum_data / count_data  
        
        group_means.append(mean_data)  
        group_counts.append(count_data)  
        grand_mean_list.extend(data)  

        total_samples += count_data  

    if num_groups < 2:  
        return {'F-statistic': None, 'P-value': None}  

    grand_mean = sum(grand_mean_list) / len(grand_mean_list)  

    # Calculate variabilities  
    ss_between = sum(count * ((mean - grand_mean) ** 2) for mean, count in zip(group_means, group_counts))  
    ss_within = sum(sum((score - mean) ** 2 for score in scores_data[school]) for school, mean in zip(scores_data, group_means))  

    df_between = num_groups - 1  
    df_within = total_samples - num_groups  

    ms_between = ss_between / df_between  
    ms_within = ss_within / df_within  

    # Compute F-statistic  
    F_statistic = ms_between / ms_within  

    # Print the calculated F-statistic value for debugging  
    print(f"Calculated F-statistic value: {F_statistic}")  

    return {'F-statistic': round(F_statistic, 2), 'P-value': None}  

def main():  
    # Example scores data  
    scores_data = {  
        'school1': [82, 85, 88, 90, 95],  
        'school2': [78, 80, 85, 88, 92],  
        'school3': [75, 79, 82, 86, 90],  
        'school4': [70, 75, 80, 85, 88]  
    }  

    # Perform ANOVA test  
    result = anova_test_exam_scores(scores_data)  
    
    # Print result  
    print(result)  # Expected output: {'F-statistic': ..., 'P-value': None}  

if __name__ == "__main__":  
    main()