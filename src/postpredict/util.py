# I got this code from AI
import numpy as np


def argsort_random_tiebreak(arr: np.ndarray,
                            rng: np.random.Generator) -> np.ndarray:
    # Generate random values for each element
    random_values = rng.random(len(arr))
    
    # Create a structured array to sort by the original values first, then the random values
    structured_array = np.array(list(zip(arr, random_values)), dtype=[("values", arr.dtype), ("random", "float64")])
    
    # Sort the structured array
    sorted_indices = np.argsort(structured_array, order=("values", "random"))
    
    return sorted_indices
