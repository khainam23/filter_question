# Calculate fitness based only on difficulty
import numpy as np

def fitness(position, matrix, questions, required_difficulty):
    """
    Fitness function with one parameter (difficulty)
    
    Args:
        position: Binary position vector indicating selected questions
        matrix: Correlation matrix between questions
        questions: Dictionary of questions by CP
        required_difficulty: Target difficulty level
        
    Returns:
        fitness_value: The fitness value to be minimized
    """
    # Convert position to binary (0 or 1)
    binary_position = np.where(position > 0.5, 1, 0)
    
    # Get selected questions
    selected_questions = []
    question_index = 0
    
    for cp in matrix.keys():
        for q in questions[cp]:
            if binary_position[question_index] == 1:
                selected_questions.append(q)
            question_index += 1
    
    # If no questions selected, return a high fitness value
    if len(selected_questions) == 0:
        return 1000.0
    
    # Calculate average difficulty of selected questions
    total_difficulty = sum(q['DL'] for q in selected_questions)
    avg_difficulty = total_difficulty / len(selected_questions)
    
    # Calculate fitness as the absolute difference between average difficulty and required difficulty
    fitness_value = abs(avg_difficulty - required_difficulty)
    
    return fitness_value