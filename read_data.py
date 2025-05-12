import csv
from collections import defaultdict

def read_matrix(file_path='matrix.csv'):
    """
    Read matrix data from a CSV file.
    
    Parameters:
    file_path (str): Path to the matrix CSV file. Default is 'matrix.csv'.
    
    Returns:
    dict: A dictionary where keys are CP values and values are N values.
    """
    matrix_data = {}
    
    try:
        with open(file_path, 'r') as file:
            # Skip the header row
            header = file.readline()
            print(f"Matrix file header: {header.strip()}")
            
            for line in file:
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                    
                print(f"Processing line: {line}")
                
                try:
                    # Split by any whitespace (spaces or tabs) and filter out empty strings
                    parts = [part for part in line.split() if part]
                    
                    if len(parts) >= 2:
                        cp = int(parts[0])
                        n = int(parts[1])
                        
                        matrix_data[cp] = n
                        print(f"Added CP: {cp}, N: {n}")
                    else:
                        print(f"Skipping line with insufficient data: {line}")
                        
                except ValueError as ve:
                    print(f"Error converting values in line '{line}': {ve}")
    
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"Error reading matrix file: {e}")
    
    print(f"Matrix data loaded: {matrix_data}")
    return matrix_data

def read_question(file_path='question.csv'):
    """
    Read question data from a CSV file.
    
    Parameters:
    file_path (str): Path to the question CSV file. Default is 'question.csv'.
    
    Returns:
    dict: A dictionary where keys are CP values and values are lists of question dictionaries.
    """
    question_entry = defaultdict(list)  # Tự động tạo list khi truy cập key mới
    
    try:
        with open(file_path, 'r') as file:
            # Skip the header row
            header = file.readline()
            print(f"Question file header: {header.strip()}")
            
            row_count = 0
            for line in file:
                row_count += 1
                line = line.strip()
                
                if not line:  # Skip empty lines
                    continue
                    
                if row_count <= 5:  # Print first few rows for debugging
                    print(f"Processing question line: {line}")
                
                try:
                    # Split by any whitespace (spaces or tabs) and filter out empty strings
                    parts = [part for part in line.split() if part]
                    
                    if len(parts) >= 4:
                        cq = int(parts[0])
                        cp = int(parts[1])
                        tq = float(parts[2])
                        dl = float(parts[3])
                        
                        question_entry[cp].append({
                            'CQ': cq,      # Mã câu hỏi
                            'CP': cp,      # Mã chương trình (nhóm câu hỏi)
                            'TQ': tq,      # Thời gian hoàn thành bài
                            'DL': dl       # Độ khó câu hỏi
                        })
                    else:
                        if row_count <= 5:
                            print(f"Skipping line with insufficient data: {line}")
                            
                except (ValueError, IndexError) as e:
                    if row_count <= 5:
                        print(f"Error processing line '{line}': {e}")

            print(f"Total questions loaded: {sum(len(questions) for questions in question_entry.values())}")
            print(f"Questions by CP: {', '.join([f'CP {cp}: {len(questions)}' for cp, questions in question_entry.items()])}")

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"Error reading question file: {e}")
    
    return question_entry