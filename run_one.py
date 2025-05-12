import read_data
import numpy as np
from sho import SpottedHyenaOptimizer
import csv

# Đọc thông tin dự án
# Task cần thực hiện - Đọc ma trận yêu cầu
matrix = read_data.read_matrix()

# Đọc dữ liệu câu hỏi
questions = read_data.read_question()

# Tính toán tổng số câu hỏi
total_questions = sum(len(questions[cp]) for cp in questions)
print(f"Tổng số câu hỏi: {total_questions}")
print(f"Số nhóm câu hỏi (CP): {len(questions)}")

# In thông tin về ma trận yêu cầu
print("\nYêu cầu số lượng câu hỏi cho mỗi CP:")
for cp, n in matrix.items():
    if cp in questions:
        print(f"CP {cp}: {n}/{len(questions[cp])} câu hỏi")
    else:
        print(f"CP {cp}: {n}/0 câu hỏi (không có dữ liệu)")

# Khởi tạo tham số cho thuật toán SHO
num_hyenas = 30  # Số lượng hyena
max_iter = 100   # Số lần lặp tối đa
dim = total_questions  # Số chiều (tổng số câu hỏi)
lb = 0  # Giới hạn dưới
ub = 1  # Giới hạn trên

# Khởi chạy thuật toán SHO
print("Bắt đầu tối ưu hóa với thuật toán Spotted Hyena Optimizer...")
sho = SpottedHyenaOptimizer(num_hyenas, max_iter, lb, ub, matrix, questions)
best_position, best_fitness = sho.optimize()

# Lấy danh sách câu hỏi được chọn
selected_questions = sho.get_selected_questions()

print(f"\nKết quả tối ưu hóa:")
print(f"Tổng số câu hỏi: {len(selected_questions)}")
print(f"Giá trị fitness tốt nhất: {best_fitness}")

# Phân tích kết quả
cp_counts = {}
total_difficulty = 0
min_difficulty = 1.0
max_difficulty = 0.0
total_time = 0  # Tổng thời gian
min_time = float('inf')
max_time = 0.0

for question in selected_questions:
    cp = question['CP']
    difficulty = question['DL']  # Độ khó câu hỏi
    time = question['TQ']      # Thời gian hoàn thành
    
    # Đếm số câu hỏi theo CP
    if cp in cp_counts:
        cp_counts[cp] += 1
    else:
        cp_counts[cp] = 1
    
    # Tính toán độ khó
    total_difficulty += difficulty
    min_difficulty = min(min_difficulty, difficulty)
    max_difficulty = max(max_difficulty, difficulty)
    
    # Tính toán thời gian
    total_time += time
    min_time = min(min_time, time)
    max_time = max(max_time, time)

# Tính giá trị trung bình
avg_difficulty = total_difficulty / len(selected_questions) if selected_questions else 0
avg_time = total_time / len(selected_questions) if selected_questions else 0

# In thống kê theo CP
print("\nThống kê theo CP:")
for cp, count in cp_counts.items():
    print(f"CP {cp}: {count} câu hỏi")

# In thống kê về độ khó
print(f"\nThống kê về độ khó:")
print(f"Độ khó trung bình: {avg_difficulty:.4f}")
print(f"Độ khó thấp nhất: {min_difficulty:.4f}")
print(f"Độ khó cao nhất: {max_difficulty:.4f}")

# In thống kê về thời gian
print(f"\nThống kê về thời gian:")
print(f"Thời gian trung bình: {avg_time:.4f}")
print(f"Thời gian thấp nhất: {min_time:.4f}")
print(f"Thời gian cao nhất: {max_time:.4f}")

# Tính tổng thời gian của đề thi
total_exam_time = sum(question['TQ'] for question in selected_questions)
print(f"\nTổng thời gian dự kiến của đề thi: {total_exam_time:.2f}")

# Lưu kết quả vào file CSV
with open('selected_questions.csv', 'w', newline='') as file:
    writer = csv.writer(file, delimiter='\t')
    writer.writerow(['CQ', 'CP', 'TQ', 'DL'])
    
    for question in selected_questions:
        writer.writerow([question['CQ'], question['CP'], question['TQ'], question['DL']])

print("\nĐã lưu kết quả vào file selected_questions.csv")
