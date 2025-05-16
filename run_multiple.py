import sys
import os

# Thêm thư mục hiện tại vào sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import read_data
import numpy as np
from sho import SpottedHyenaOptimizer
import csv
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# Tạo thư mục để lưu kết quả
results_dir = "multiple_runs_results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Tạo thư mục con với timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = os.path.join(results_dir, f"run_{timestamp}")
os.makedirs(run_dir)

# Number of runs
num_runs = 1
difficulty_values = np.arange(0.1, 1.0, 0.1)  # Difficulty values from 0.1 to 0.9

# Fixed parameters for all runs
# num_questions = 100        # Number of questions in each exam

# Khởi tạo dictionary để lưu kết quả theo difficulty
results_by_difficulty = {}
for difficulty in difficulty_values:
    results_by_difficulty[difficulty] = {
        'all_results': [],
        'best_fitness_values': [],
        'best_run_info': None,
        'best_fitness': float('inf'),
        'worst_run_info': None,
        'worst_fitness': 0,
        'total_time': 0
    }

print(f"Bắt đầu chạy thuật toán với {len(difficulty_values)} giá trị độ khó, mỗi giá trị chạy {num_runs} lần...")

for difficulty in difficulty_values:
    print(f"\n=== Chạy với độ khó yêu cầu = {difficulty:.1f} ===")
    
    for run in range(1, num_runs + 1):
        print(f"\nLần chạy thứ {run}/{num_runs}")
        
        # Đọc dữ liệu
        matrix = read_data.read_matrix('1200_matrix.csv')
        questions = read_data.read_question('1200_question.csv')
        
        # Tính toán tổng số câu hỏi
        total_questions = sum(len(questions[cp]) for cp in questions)
        
        # Initialize parameters for SHO algorithm with current difficulty
        num_hyenas = 30
        max_iter = 100
        lb = 0
        ub = 1
        
        # Record start time
        start_time = time.time()
        
        # Run SHO algorithm with specified difficulty (no alpha needed)
        sho = SpottedHyenaOptimizer(num_hyenas, max_iter, lb, ub, matrix, questions, required_difficulty=difficulty)
        best_position, run_fitness = sho.optimize()
        
        # Ghi lại thời gian kết thúc
        end_time = time.time()
        run_time = end_time - start_time
        results_by_difficulty[difficulty]['total_time'] += run_time
        
        # Lấy danh sách câu hỏi được chọn
        selected_questions = sho.get_selected_questions()
        
        # Phân tích kết quả
        cp_counts = {}
        total_difficulty = 0
        min_difficulty = 1.0
        max_difficulty = 0.0
        total_time_questions = 0
        min_time = float('inf')
        max_time = 0.0
        
        for question in selected_questions:
            cp = question['CP']
            question_difficulty = question['DL']
            time_q = question['TQ']
            
            # Đếm số câu hỏi theo CP
            if cp in cp_counts:
                cp_counts[cp] += 1
            else:
                cp_counts[cp] = 1
            
            # Tính toán độ khó
            total_difficulty += question_difficulty
            min_difficulty = min(min_difficulty, question_difficulty)
            max_difficulty = max(max_difficulty, question_difficulty)
            
            # Tính toán thời gian
            total_time_questions += time_q
            min_time = min(min_time, time_q)
            max_time = max(max_time, time_q)
        
        # Tính giá trị trung bình
        avg_difficulty = total_difficulty / len(selected_questions) if selected_questions else 0
        avg_time = total_time_questions / len(selected_questions) if selected_questions else 0
        
        # Tính độ lệch chuẩn
        dl_variance = sum((q['DL'] - avg_difficulty) ** 2 for q in selected_questions) / len(selected_questions) if selected_questions else 0
        tq_variance = sum((q['TQ'] - avg_time) ** 2 for q in selected_questions) / len(selected_questions) if selected_questions else 0
        dl_std = np.sqrt(dl_variance)
        tq_std = np.sqrt(tq_variance)
        
        # Lưu thông tin về lần chạy này
        run_info = {
            'run': run,
            'fitness': run_fitness,
            'num_questions': len(selected_questions),
            'avg_difficulty': avg_difficulty,
            'min_difficulty': min_difficulty,
            'max_difficulty': max_difficulty,
            'dl_std': dl_std,
            'avg_time': avg_time,
            'min_time': min_time,
            'max_time': max_time,
            'tq_std': tq_std,
            'total_time_questions': total_time_questions,
            'execution_time': run_time
        }
        
        # Thêm thông tin về số lượng câu hỏi cho mỗi CP
        for cp in matrix.keys():
            run_info[f'CP_{cp}'] = cp_counts.get(cp, 0)
        
        results_by_difficulty[difficulty]['all_results'].append(run_info)
        results_by_difficulty[difficulty]['best_fitness_values'].append(run_fitness)
        
        # Kiểm tra xem đây có phải là lần chạy tốt nhất không
        if run_fitness < results_by_difficulty[difficulty]['best_fitness']:
            results_by_difficulty[difficulty]['best_fitness'] = run_fitness
            results_by_difficulty[difficulty]['best_run_info'] = run_info.copy()
            
            # Lưu kết quả của lần chạy tốt nhất cho mỗi difficulty
            best_run_file = os.path.join(run_dir, f"best_run_difficulty_{difficulty:.1f}_run_{run}.csv")
            with open(best_run_file, 'w', newline='') as file:
                writer = csv.writer(file, delimiter='\t')
                writer.writerow(['CQ', 'CP', 'TQ', 'DL'])
                for question in selected_questions:
                    writer.writerow([question['CQ'], question['CP'], question['TQ'], question['DL']])
        
        # Kiểm tra xem đây có phải là lần chạy tệ nhất không
        if run_fitness > results_by_difficulty[difficulty]['worst_fitness']:
            results_by_difficulty[difficulty]['worst_fitness'] = run_fitness
            results_by_difficulty[difficulty]['worst_run_info'] = run_info.copy()

# Create summary DataFrame for all difficulty values
summary_results = []
for difficulty in difficulty_values:
    results = results_by_difficulty[difficulty]
    avg_fitness = np.mean(results['best_fitness_values'])
    std_fitness = np.std(results['best_fitness_values'])
    avg_time = results['total_time'] / num_runs
    
    # Calculate average difficulty and time for best runs
    if results['best_run_info']:
        best_avg_difficulty = results['best_run_info']['avg_difficulty']
        best_total_time = results['best_run_info']['total_time_questions']
        difficulty_deviation = abs(best_avg_difficulty - difficulty)  # Deviation from target difficulty
        time_deviation = results['best_run_info']['avg_time']  # Just record average time per question
    else:
        best_avg_difficulty = 0
        best_total_time = 0
        difficulty_deviation = 1
        time_deviation = 0
    
    summary_results.append({
        'difficulty': difficulty,
        'avg_fitness': avg_fitness,
        'std_fitness': std_fitness,
        'best_fitness': results['best_fitness'],
        'worst_fitness': results['worst_fitness'],
        'avg_time': avg_time,
        'best_avg_difficulty': best_avg_difficulty,
        'best_total_time': best_total_time,
        'difficulty_deviation': difficulty_deviation,
        'avg_time_per_question': time_deviation
    })

# Lưu kết quả tổng hợp
summary_df = pd.DataFrame(summary_results)
summary_df.to_csv(os.path.join(run_dir, "difficulty_summary_results.csv"), index=False)

# Create comparison charts by difficulty
plt.figure(figsize=(12, 6))
plt.errorbar(difficulty_values, summary_df['avg_fitness'], 
            yerr=summary_df['std_fitness'], 
            fmt='o-', capsize=5)
plt.title('Average Fitness by Target Difficulty Value')
plt.xlabel('Target Difficulty')
plt.ylabel('Average Fitness (± Standard Deviation)')
plt.grid(True)
plt.savefig(os.path.join(run_dir, 'difficulty_comparison.png'), dpi=300, bbox_inches='tight')

# Create a more detailed explanation of what difficulty represents
plt.figure(figsize=(14, 8))
plt.subplot(2, 1, 1)
plt.plot(difficulty_values, summary_df['avg_fitness'], 'o-', color='blue', linewidth=2)
plt.fill_between(difficulty_values, 
                summary_df['avg_fitness'] - summary_df['std_fitness'],
                summary_df['avg_fitness'] + summary_df['std_fitness'],
                alpha=0.2, color='blue')
plt.title('Effect of Target Difficulty on Fitness Value', fontsize=14)
plt.ylabel('Fitness Value', fontsize=12)
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(difficulty_values, summary_df['avg_time'], 'o-', color='green', linewidth=2)
plt.title('Effect of Target Difficulty on Algorithm Runtime', fontsize=14)
plt.xlabel('Target Difficulty', fontsize=12)
plt.ylabel('Average Runtime (seconds)', fontsize=12)
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(run_dir, 'difficulty_detailed_analysis.png'), dpi=300, bbox_inches='tight')

# Create a chart showing actual difficulty vs target difficulty
plt.figure(figsize=(14, 8))
plt.subplot(2, 1, 1)
plt.plot(difficulty_values, summary_df['best_avg_difficulty'], 'o-', color='red', linewidth=2, label='Actual Difficulty')
plt.plot(difficulty_values, difficulty_values, '--', color='gray', linewidth=1, label='Target Difficulty')
plt.title('Actual vs Target Difficulty', fontsize=14)
plt.ylabel('Difficulty', fontsize=12)
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(difficulty_values, summary_df['difficulty_deviation'], 'o-', color='purple', linewidth=2, label='Difficulty Deviation')
plt.xlabel('Target Difficulty', fontsize=12)
plt.ylabel('Absolute Deviation from Target', fontsize=12)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(run_dir, 'difficulty_deviations.png'), dpi=300, bbox_inches='tight')

# Print summary of results
print("\nResults by difficulty have been saved to:", run_dir)
print("\nSummary of results:")
print("------------------")
print(f"{'Difficulty':<10} {'Avg Fitness':<15} {'Best Fitness':<15} {'Actual Diff':<15} {'Deviation':<15}")
print("-" * 70)

for _, row in summary_df.iterrows():
    print(f"{row['difficulty']:<10.1f} {row['avg_fitness']:<15.6f} {row['best_fitness']:<15.6f} {row['best_avg_difficulty']:<15.6f} {row['difficulty_deviation']:<15.6f}")

# Find the best difficulty based on fitness
best_difficulty_idx = summary_df['avg_fitness'].idxmin()
best_difficulty = summary_df.iloc[best_difficulty_idx]['difficulty']

print("\nBest performing target difficulty:", best_difficulty)
print(f"This difficulty value resulted in the lowest fitness value (best match between target and actual difficulty)")
print(f"Results and charts have been saved to: {run_dir}")