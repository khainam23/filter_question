import numpy as np
import random
import math
from fitness_1w import fitness

class SpottedHyenaOptimizer:

    def __init__(self, num_hyenas, max_iter, lb, ub, matrix, questions, required_difficulty=0.5):
        """
        Initialize the Spotted Hyena Optimizer
        
        Parameters:
        num_hyenas: Number of search agents (hyenas)
        max_iter: Maximum number of iterations
        lb: Lower bound of variables
        ub: Upper bound of variables
        matrix: Dictionary containing CP (key) and N (value) pairs
        questions: Dictionary containing questions grouped by CP
        required_difficulty: Target difficulty level for the exam
        """
        self.num_hyenas = num_hyenas
        self.max_iter = max_iter
        self.lb = lb
        self.ub = ub
        self.matrix = matrix
        self.questions = questions
        self.required_difficulty = required_difficulty
        
        # Check if there are any questions available for the given CPs in the matrix
        available_questions_count = 0
        for cp in self.matrix:
            if cp in self.questions:
                available_questions_count += len(self.questions[cp])
        
        if available_questions_count == 0:
            print("Error: No questions found for the CPs in the matrix. Check if CP values in matrix match those in questions file.")
            # Create a default matrix using available CP values from questions
            self.matrix = {}
            for cp in self.questions:
                # Distribute questions evenly among available CPs
                self.matrix[cp] = 10  # Default value
            
            # Adjust to ensure total is 100
            total_cps = len(self.matrix)
            if total_cps > 0:
                base_count = 100 // total_cps
                remainder = 100 % total_cps
                
                for cp in self.matrix:
                    self.matrix[cp] = base_count
                
                # Distribute the remainder
                for cp in list(self.matrix.keys())[:remainder]:
                    self.matrix[cp] += 1
                
                print(f"Created a new matrix with available CPs: {self.matrix}")
            else:
                print("Error: No valid CPs found in questions file.")
                # Set a dummy matrix to prevent further errors
                self.matrix = {1: 100}
        
        # Ensure the matrix sums to 100 questions
        total_questions = sum(self.matrix.values())
        if total_questions != 100:
            print(f"Warning: Matrix requires {total_questions} questions, but we need exactly 100.")
            # Adjust the matrix to have exactly 100 questions
            if total_questions > 0:  # Prevent division by zero
                scale_factor = 100 / total_questions
                for cp in self.matrix:
                    self.matrix[cp] = round(self.matrix[cp] * scale_factor)
                
                # Make final adjustments to ensure exactly 100 questions
                total = sum(self.matrix.values())
                if total < 100:
                    # Add the remaining questions to the first CP
                    first_cp = list(self.matrix.keys())[0]
                    self.matrix[first_cp] += (100 - total)
                elif total > 100:
                    # Remove excess questions from the first CP with more than 1 question
                    for cp in self.matrix:
                        if self.matrix[cp] > 1:
                            self.matrix[cp] -= (total - 100)
                            break
                
                print(f"Adjusted matrix to have exactly 100 questions: {self.matrix}")
            else:
                print("Error: Cannot adjust matrix with zero questions.")
                # Set a default matrix
                if self.questions:
                    first_cp = list(self.questions.keys())[0]
                    self.matrix = {first_cp: 100}
                    print(f"Set default matrix to use CP {first_cp} for all 100 questions.")
        
        # Create mapping from position in vector to questions
        self.position_mapping = []
        for cp in self.questions:
            for i in range(len(self.questions[cp])):
                self.position_mapping.append((cp, i))
        
        # Ensure the dimension of position vector matches the number of questions
        self.dim = len(self.position_mapping)
        
        # Initialize positions of hyenas
        self.positions = np.zeros((num_hyenas, self.dim))
        for i in range(num_hyenas):
            # Initialize randomly but ensure constraints on number of questions
            self.positions[i] = self.initialize_position()
        
        # Initialize best solution
        self.best_position = np.zeros(self.dim)
        self.best_fitness = float('inf')
    
    def initialize_position(self):
        """
        Initialize a valid position that satisfies the constraints
        Each hyena carries the complete information of the matrix
        
        Returns:
        position: A valid position vector
        """
        position = np.zeros(self.dim)
        
        # For each CP, randomly select N questions as required by the matrix
        for cp in self.matrix:
            if cp not in self.questions:
                continue
                
            required_n = self.matrix[cp]
            cp_questions = self.questions[cp]
            
            # If there aren't enough questions, select all of them
            if len(cp_questions) <= required_n:
                indices = list(range(len(cp_questions)))
            else:
                # Randomly select N questions
                indices = random.sample(range(len(cp_questions)), required_n)
            
            # Mark the selected questions
            for idx in indices:
                # Find the position in the vector
                for pos_idx, (map_cp, map_idx) in enumerate(self.position_mapping):
                    if map_cp == cp and map_idx == idx:
                        position[pos_idx] = 1
                        break
        
        # Verify that exactly 100 questions are selected
        selected_count = np.sum(position)
        if selected_count != 100:
            print(f"Warning: Initialized position has {selected_count} questions instead of 100.")
        
        return position
        
    def evaluate_fitness(self, position):
        """
        Evaluate fitness of a position using the formula
        
        Parameters:
        position: Binary position vector (0 or 1)
        
        Returns:
        fitness_value: The fitness value of the position
        """
        # Convert position to binary (0 or 1)
        binary_position = np.round(position).astype(int)
        
        # Check constraints for each CP
        cp_counts = {}
        selected_questions = []
        
        # Count selected questions for each CP
        for pos_idx, is_selected in enumerate(binary_position):
            if is_selected == 1 and pos_idx < len(self.position_mapping):
                cp, q_idx = self.position_mapping[pos_idx]
                
                # Count questions for this CP
                if cp in cp_counts:
                    cp_counts[cp] += 1
                else:
                    cp_counts[cp] = 1
                
                # Add question to the list
                selected_questions.append(self.questions[cp][q_idx])
        
        # Check if the number of questions for each CP is correct
        valid = True
        for cp, required_n in self.matrix.items():
            if cp in cp_counts and cp_counts[cp] != required_n:
                valid = False
                break
            elif cp not in cp_counts and required_n > 0:
                valid = False
                break
        
        # If constraints are not satisfied, return a large penalty value
        if not valid or not selected_questions:
            return float('inf')
        
        # Check if we have exactly 100 questions as required
        if len(selected_questions) != 100:
            return float('inf')
        
        # Use our new fitness function with the position vector, matrix, questions, and required difficulty
        fitness_value = fitness(binary_position, self.matrix, self.questions, self.required_difficulty)
        
        return fitness_value
    
    def update_position(self, current_pos, best_pos, B, E):
        """
        Update the position of a hyena
        
        Parameters:
        current_pos: Current position of the hyena
        best_pos: Best position found so far
        a: Parameter that decreases linearly from 2 to 0
        C: Random vector in [0, 2]
        M: Random vector in [0, 1]
        
        Returns:
        new_pos: Updated position of the hyena
        """
        # Cập nhật vị trí theo thuật toán SHO
        D = abs(B * best_pos - current_pos)
        new_pos = best_pos - D * E
        
        # Đảm bảo vị trí nằm trong giới hạn
        new_pos = np.clip(new_pos, self.lb, self.ub)
        
        return new_pos
    
    def optimize(self):
        """
        Main optimization loop
        
        Returns:
        best_position: Best position found
        best_fitness: Best fitness value
        """
        # Khởi tạo bộ đếm lặp
        t = 0
        
        # Đánh giá vị trí ban đầu
        for i in range(self.num_hyenas):
            fitness_value = self.evaluate_fitness(self.positions[i])
            
            # Cập nhật giải pháp tốt nhất nếu tốt hơn
            if fitness_value < self.best_fitness:
                self.best_fitness = fitness_value
                self.best_position = self.positions[i].copy()
        
        # Nếu không tìm thấy giải pháp hợp lệ, tạo thêm vị trí ngẫu nhiên
        if self.best_fitness == float('inf'):
            print("Khởi tạo thêm vị trí ngẫu nhiên để tìm giải pháp hợp lệ...")
            for _ in range(200):  # Thử thêm 200 vị trí ngẫu nhiên
                pos = self.initialize_position()
                fitness_value = self.evaluate_fitness(pos)
                
                if fitness_value < self.best_fitness:
                    self.best_fitness = fitness_value
                    self.best_position = pos.copy()
                    
                    # Thay thế hyena tệ nhất bằng vị trí mới này
                    worst_idx = 0
                    worst_fitness = self.evaluate_fitness(self.positions[0])
                    
                    for i in range(1, self.num_hyenas):
                        current_fitness = self.evaluate_fitness(self.positions[i])
                        if current_fitness > worst_fitness:
                            worst_fitness = current_fitness
                            worst_idx = i
                    
                    self.positions[worst_idx] = pos.copy()
        
        # Lưu trữ lịch sử fitness tốt nhất để theo dõi sự hội tụ
        fitness_history = []
        stagnation_counter = 0
        
        while t < self.max_iter:
            # Cập nhật tham số a (giảm tuyến tính từ 5 đến 0)
            a = 5 - t * (5 / self.max_iter)
            
            # Cập nhật vị trí của tất cả hyenas
            for i in range(self.num_hyenas):
                # Tạo vector ngẫu nhiên
                B = 2 * np.random.rand()
                E = 2 * a * np.random.rand() - a
                
                # Cập nhật vị trí
                new_pos = self.update_position(self.positions[i], self.best_position, B, E)
                
                # Đảm bảo giá trị nhị phân (0 hoặc 1) bằng hàm sigmoid
                # Sau khi cập nhật vị trí, cần đảm bảo rằng mỗi hyena vẫn mang đủ thông tin của matrix
                binary_pos = np.zeros(self.dim)
                for j in range(self.dim):
                    sigmoid_val = 1 / (1 + math.exp(-10 * (new_pos[j] - 0.5)))
                    if np.random.rand() < sigmoid_val:
                        binary_pos[j] = 1
                    else:
                        binary_pos[j] = 0
                
                # Kiểm tra xem vị trí mới có thỏa mãn ràng buộc không
                cp_counts = {}
                for pos_idx, is_selected in enumerate(binary_pos):
                    if is_selected == 1 and pos_idx < len(self.position_mapping):
                        cp, _ = self.position_mapping[pos_idx]
                        if cp in cp_counts:
                            cp_counts[cp] += 1
                        else:
                            cp_counts[cp] = 1
                
                # Nếu không thỏa mãn ràng buộc, khởi tạo lại vị trí
                valid = True
                for cp, required_n in self.matrix.items():
                    if cp in cp_counts and cp_counts[cp] != required_n:
                        valid = False
                        break
                    elif cp not in cp_counts and required_n > 0:
                        valid = False
                        break
                
                if not valid:
                    new_pos = self.initialize_position()
                else:
                    new_pos = binary_pos
                
                # Đánh giá vị trí mới
                new_fitness = self.evaluate_fitness(new_pos)
                
                # Cập nhật vị trí nếu tốt hơn
                if new_fitness < self.evaluate_fitness(self.positions[i]):
                    self.positions[i] = new_pos.copy()
                
                # Cập nhật giải pháp tốt nhất nếu tốt hơn
                if new_fitness < self.best_fitness:
                    self.best_fitness = new_fitness
                    self.best_position = new_pos.copy()
                    stagnation_counter = 0  # Reset stagnation counter
                
            # Tăng bộ đếm lặp
            t += 1
            
            # Lưu trữ fitness tốt nhất hiện tại
            fitness_history.append(self.best_fitness)
            
            # Kiểm tra sự hội tụ
            if len(fitness_history) > 10:
                # Nếu fitness không cải thiện trong 10 lần lặp gần nhất
                if all(abs(fitness_history[-1] - f) < 1e-6 for f in fitness_history[-10:]):
                    stagnation_counter += 1
                else:
                    stagnation_counter = 0
            
            # Nếu thuật toán bị mắc kẹt trong tối ưu cục bộ, thực hiện đột biến
            if stagnation_counter >= 5:
                print(f"Iteration {t}/{self.max_iter}, Stagnation detected, performing mutation...")
                
                # Đột biến một số hyena
                for i in range(self.num_hyenas // 2):  # Đột biến một nửa quần thể
                    # Tạo vị trí mới hoàn toàn
                    self.positions[i] = self.initialize_position()
                
                stagnation_counter = 0
            
            # In tiến trình
            if t % 10 == 0:
                print(f"Iteration {t}/{self.max_iter}, Best Fitness: {self.best_fitness}")
                
                # Nếu không tìm thấy giải pháp hợp lệ sau 50 lần lặp, thử khởi tạo lại
                if t >= 50 and self.best_fitness == float('inf'):
                    print("Không tìm thấy giải pháp hợp lệ, thử khởi tạo lại...")
                    for i in range(self.num_hyenas):
                        self.positions[i] = self.initialize_position()
        
        # Nếu vẫn không tìm thấy giải pháp hợp lệ, tạo một giải pháp đơn giản
        if self.best_fitness == float('inf'):
            print("Không tìm thấy giải pháp tối ưu, tạo giải pháp đơn giản...")
            self.best_position = self.initialize_position()
            self.best_fitness = self.evaluate_fitness(self.best_position)
        
        # Thực hiện tìm kiếm cục bộ để cải thiện giải pháp tốt nhất
        print("Thực hiện tìm kiếm cục bộ để cải thiện giải pháp...")
        self.local_search()
        
        return self.best_position, self.best_fitness
        
    def local_search(self):
        """
        Thực hiện tìm kiếm cục bộ để cải thiện giải pháp tốt nhất
        """
        improved = True
        iterations = 0
        max_iterations = 100
        
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            
            # Tạo bản sao của vị trí tốt nhất
            current_position = self.best_position.copy()
            current_fitness = self.best_fitness
            
            # Thử đảo bit từng vị trí
            for i in range(self.dim):
                # Đảo bit tại vị trí i
                new_position = current_position.copy()
                new_position[i] = 1 - new_position[i]
                
                # Kiểm tra xem vị trí mới có thỏa mãn ràng buộc không
                cp_counts = {}
                for pos_idx, is_selected in enumerate(new_position):
                    if is_selected == 1 and pos_idx < len(self.position_mapping):
                        cp, _ = self.position_mapping[pos_idx]
                        if cp in cp_counts:
                            cp_counts[cp] += 1
                        else:
                            cp_counts[cp] = 1
                
                # Kiểm tra ràng buộc
                valid = True
                for cp, required_n in self.matrix.items():
                    if cp in cp_counts and cp_counts[cp] != required_n:
                        valid = False
                        break
                    elif cp not in cp_counts and required_n > 0:
                        valid = False
                        break
                
                # Chỉ đánh giá nếu vị trí mới hợp lệ
                if valid:
                    # Đánh giá vị trí mới
                    new_fitness = self.evaluate_fitness(new_position)
                    
                    # Nếu vị trí mới tốt hơn, cập nhật
                    if new_fitness < current_fitness:
                        current_position = new_position.copy()
                        current_fitness = new_fitness
                        improved = True
            
            # Cập nhật giải pháp tốt nhất
            if current_fitness < self.best_fitness:
                self.best_position = current_position.copy()
                self.best_fitness = current_fitness
                print(f"Local search iteration {iterations}, Improved fitness: {self.best_fitness}")
            
        print(f"Local search completed after {iterations} iterations, Final fitness: {self.best_fitness}")
    
    def get_selected_questions(self):
        """
        Get the selected questions based on the best position
        
        Returns:
        selected_questions: List of selected questions
        """
        binary_position = np.round(self.best_position).astype(int)
        selected_questions = []
        
        # Get the selected questions based on the best position
        for pos_idx, is_selected in enumerate(binary_position):
            if is_selected == 1 and pos_idx < len(self.position_mapping):
                cp, q_idx = self.position_mapping[pos_idx]
                
                # Add the question to the list
                if cp in self.questions and q_idx < len(self.questions[cp]):
                    question = self.questions[cp][q_idx].copy()  # Create a copy to avoid modifying the original
                    selected_questions.append(question)
        
        # Verify that the result satisfies the matrix requirements
        cp_counts = {}
        for question in selected_questions:
            cp = question['CP']
            if cp in cp_counts:
                cp_counts[cp] += 1
            else:
                cp_counts[cp] = 1
        
        # Print verification information
        print("\nResult verification:")
        for cp, required_n in self.matrix.items():
            actual_n = cp_counts.get(cp, 0)
            print(f"CP {cp}: Required {required_n}, Actual {actual_n}")
        
        # Calculate and print fitness metrics
        total_difficulty = sum(q['DL'] for q in selected_questions)
        total_time = sum(q['TQ'] for q in selected_questions)
        avg_difficulty = total_difficulty / len(selected_questions) if selected_questions else 0
        
        print(f"\nTotal questions: {len(selected_questions)}")
        print(f"Average difficulty: {avg_difficulty:.4f} (Target: {self.required_difficulty:.1f})")
        print(f"Total time: {total_time:.2f} (Target: 5400)")
        
        # Create binary position vector for the selected questions
        binary_position = np.zeros(self.dim)
        for pos_idx, (cp, q_idx) in enumerate(self.position_mapping):
            for question in selected_questions:
                if question['CP'] == cp and question['CQ'] == self.questions[cp][q_idx]['CQ']:
                    binary_position[pos_idx] = 1
                    break
        
        # Calculate fitness value using our new fitness function
        fitness_value = fitness(binary_position, self.matrix, self.questions, self.required_difficulty)
        print(f"Fitness value: {fitness_value:.6f} (target difficulty={self.required_difficulty})")
        
        return selected_questions