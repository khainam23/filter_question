# Calculate based on difficulty and time

def fitness(questions, alpha=0.2, required_difficulty=0.5, required_time=5400):
    """
    Calculate fitness value based on the new formula
    
    Parameters:
    questions: List of questions in the exam
    alpha: Weight for difficulty component (default is 0.2)
    required_difficulty: Required difficulty (default is 0.5)
    required_time: Required time (default is 5400)
    
    Returns:
    fitness value: Weighted combination of difficulty and time components
    """
    # Calculate difficulty component
    difficulty_component = ((sum(q.Difficulty for q in questions) / len(questions)) 
                            - required_difficulty)
    
    # Calculate time component
    time_component = (1 - (sum(q.Time for q in questions) / required_time))
    
    # Combine both components with weights
    return (alpha * abs(difficulty_component)) + ((1 - alpha) * abs(time_component))