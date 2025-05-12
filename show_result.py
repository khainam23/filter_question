import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Thiết lập style cho biểu đồ
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

# Đọc dữ liệu từ file CSV (phân tách bằng tab)
df = pd.read_csv('selected_questions.csv', delimiter='\t')

# In thông tin tổng quan
print("=== THÔNG TIN TỔNG QUAN VỀ ĐỀ THI ===")
print(f"Tổng số câu hỏi: {len(df)}")
print(f"Số lượng CP: {df['CP'].nunique()}")
print("\nThống kê theo CP:")
cp_counts = df['CP'].value_counts().sort_index()
for cp, count in cp_counts.items():
    print(f"CP {cp}: {count} câu hỏi")

# Tính toán thống kê về độ khó và thời gian
print("\n=== THỐNG KÊ VỀ ĐỘ KHÓ VÀ THỜI GIAN ===")
print(f"Độ khó trung bình: {df['DL'].mean():.4f}")
print(f"Độ khó thấp nhất: {df['DL'].min():.4f}")
print(f"Độ khó cao nhất: {df['DL'].max():.4f}")
print(f"Độ lệch chuẩn của độ khó: {df['DL'].std():.4f}")

print(f"\nThời gian trung bình: {df['TQ'].mean():.4f}")
print(f"Thời gian thấp nhất: {df['TQ'].min():.4f}")
print(f"Thời gian cao nhất: {df['TQ'].max():.4f}")
print(f"Độ lệch chuẩn của thời gian: {df['TQ'].std():.4f}")
print(f"Tổng thời gian dự kiến: {df['TQ'].sum():.2f}")

# Tính toán tương quan giữa TQ và DL
correlation = df['TQ'].corr(df['DL'])
print(f"\nHệ số tương quan giữa Thời gian và Độ khó: {correlation:.4f}")

print("\n5 dòng đầu tiên của dữ liệu:")
print(df.head())

# Tạo layout cho các biểu đồ
plt.figure(figsize=(15, 12))
gs = GridSpec(3, 2, figure=plt.gcf())

# 1. Biểu đồ phân bố giá trị TQ
ax1 = plt.subplot(gs[0, 0])
sns.histplot(df['TQ'], bins=20, kde=True, color='skyblue', ax=ax1)
ax1.axvline(df['TQ'].mean(), color='red', linestyle='--', label=f'Trung bình: {df["TQ"].mean():.2f}')
ax1.set_title('Phân bố Thời gian (TQ)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Thời gian')
ax1.set_ylabel('Số lượng câu hỏi')
ax1.legend()

# 2. Biểu đồ phân bố giá trị DL
ax2 = plt.subplot(gs[0, 1])
sns.histplot(df['DL'], bins=20, kde=True, color='lightgreen', ax=ax2)
ax2.axvline(df['DL'].mean(), color='red', linestyle='--', label=f'Trung bình: {df["DL"].mean():.2f}')
ax2.set_title('Phân bố Độ khó (DL)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Độ khó')
ax2.set_ylabel('Số lượng câu hỏi')
ax2.legend()

# 3. Biểu đồ scatter giữa TQ và DL với đường hồi quy
ax3 = plt.subplot(gs[1, :])
sns.regplot(x='TQ', y='DL', data=df, scatter_kws={'alpha':0.6}, line_kws={'color':'red'}, ax=ax3)
ax3.set_title(f'Tương quan giữa Thời gian và Độ khó (r = {correlation:.4f})', fontsize=12, fontweight='bold')
ax3.set_xlabel('Thời gian (TQ)')
ax3.set_ylabel('Độ khó (DL)')

# Thêm chú thích về mối tương quan
if correlation > 0.7:
    corr_text = "Tương quan dương mạnh"
elif correlation > 0.3:
    corr_text = "Tương quan dương trung bình"
elif correlation > 0:
    corr_text = "Tương quan dương yếu"
elif correlation > -0.3:
    corr_text = "Tương quan âm yếu"
elif correlation > -0.7:
    corr_text = "Tương quan âm trung bình"
else:
    corr_text = "Tương quan âm mạnh"

ax3.annotate(corr_text, xy=(0.05, 0.95), xycoords='axes fraction', 
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

# 4. Biểu đồ boxplot cho TQ theo CP
ax4 = plt.subplot(gs[2, 0])
sns.boxplot(x='CP', y='TQ', data=df, ax=ax4)
ax4.set_title('Phân bố Thời gian theo CP', fontsize=12, fontweight='bold')
ax4.set_xlabel('CP')
ax4.set_ylabel('Thời gian (TQ)')

# 5. Biểu đồ boxplot cho DL theo CP
ax5 = plt.subplot(gs[2, 1])
sns.boxplot(x='CP', y='DL', data=df, ax=ax5)
ax5.set_title('Phân bố Độ khó theo CP', fontsize=12, fontweight='bold')
ax5.set_xlabel('CP')
ax5.set_ylabel('Độ khó (DL)')

plt.tight_layout()
plt.savefig('analysis_results.png', dpi=300, bbox_inches='tight')
plt.show()

# Tạo biểu đồ heatmap cho tương quan giữa các biến
plt.figure(figsize=(10, 8))
correlation_matrix = df.corr()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', 
            mask=mask, cbar_kws={'label': 'Hệ số tương quan'})
plt.title('Ma trận tương quan giữa các biến', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# Tạo biểu đồ tròn thể hiện phân bố câu hỏi theo CP
plt.figure(figsize=(10, 8))
cp_counts.plot.pie(autopct='%1.1f%%', startangle=90, shadow=True, 
                  explode=[0.05] * len(cp_counts), textprops={'fontsize': 12})
plt.title('Phân bố câu hỏi theo CP', fontsize=14, fontweight='bold')
plt.ylabel('')
plt.tight_layout()
plt.savefig('cp_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# Tạo biểu đồ kết hợp giữa TQ và DL với màu sắc phân biệt theo CP
plt.figure(figsize=(12, 8))
scatter = sns.scatterplot(x='TQ', y='DL', hue='CP', data=df, palette='Set1', s=100, alpha=0.7)
plt.title('Phân bố câu hỏi theo Thời gian, Độ khó và CP', fontsize=14, fontweight='bold')
plt.xlabel('Thời gian (TQ)')
plt.ylabel('Độ khó (DL)')

# Thêm đường trung bình
plt.axhline(y=df['DL'].mean(), color='gray', linestyle='--', alpha=0.7, 
           label=f'Độ khó TB: {df["DL"].mean():.2f}')
plt.axvline(x=df['TQ'].mean(), color='gray', linestyle='--', alpha=0.7,
           label=f'Thời gian TB: {df["TQ"].mean():.2f}')

# Thêm chú thích cho từng phần tư
plt.annotate('Dễ - Nhanh', xy=(0.05, 0.05), xycoords='axes fraction', fontsize=12)
plt.annotate('Dễ - Lâu', xy=(0.85, 0.05), xycoords='axes fraction', fontsize=12)
plt.annotate('Khó - Nhanh', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12)
plt.annotate('Khó - Lâu', xy=(0.85, 0.95), xycoords='axes fraction', fontsize=12)

plt.legend(title='CP', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('question_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nĐã lưu các biểu đồ phân tích vào các file PNG.")
print("Các file đã tạo: analysis_results.png, correlation_matrix.png, cp_distribution.png, question_distribution.png")