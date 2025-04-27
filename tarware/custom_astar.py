import numpy as np
import heapq

def astar_path(grid, start, goal, allow_diagonal=False):
    """
    A custom implementation of A* pathfinding algorithm to replace pyastar2d.
    
    Args:
        grid: A 2D numpy array where values > 1 are considered obstacles
        start: A tuple or array (y, x) of the starting position
        goal: A tuple or array (y, x) of the goal position
        allow_diagonal: Whether to allow diagonal movements
        
    Returns:
        A numpy array of shape (path_length, 2) containing the path from start to goal
    """
    # Convert start and goal to tuples to make them hashable
    start_tuple = tuple(map(int, start))
    goal_tuple = tuple(map(int, goal))
    
    if not (0 <= start[0] < grid.shape[0] and 0 <= start[1] < grid.shape[1]):
        print(f"Start position {start} is outside grid bounds {grid.shape}")
        return None
    
    if not (0 <= goal[0] < grid.shape[0] and 0 <= goal[1] < grid.shape[1]):
        print(f"Goal position {goal} is outside grid bounds {grid.shape}")
        return None
    
    # Check if start or goal positions are obstacles
    if np.isinf(grid[start[0], start[1]]) or np.isinf(grid[goal[0], goal[1]]):
        print(f"Start or goal position is an obstacle")
        return None
    
    # Define possible movements (4-connected or 8-connected)
    if allow_diagonal:
        neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    else:
        neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    
    # Initialize data structures
    closed_set = set()
    came_from = {}
    g_score = {start_tuple: 0}
    f_score = {start_tuple: manhattan_distance(start, goal)}
    open_set = [(f_score[start_tuple], start_tuple)]
    
    while open_set:
        _, current_tuple = heapq.heappop(open_set)
        
        if current_tuple == goal_tuple:
            # Reconstruct path
            path = []
            current_path = current_tuple
            while current_path in came_from:
                path.append(current_path)
                current_path = came_from[current_path]
            path.append(start_tuple)
            path.reverse()
            return np.array(path)
        
        closed_set.add(current_tuple)
        
        for dy, dx in neighbors:
            neighbor = (current_tuple[0] + dy, current_tuple[1] + dx)
            
            # Check if the neighbor is valid
            if not (0 <= neighbor[0] < grid.shape[0] and 0 <= neighbor[1] < grid.shape[1]):
                continue
            
            # Check if the neighbor is an obstacle
            if np.isinf(grid[neighbor[0], neighbor[1]]):
                continue
            
            # Check if the neighbor is already evaluated
            if neighbor in closed_set:
                continue
            
            # Calculate tentative g_score
            movement_cost = grid[neighbor[0], neighbor[1]]
            if allow_diagonal and abs(dy) + abs(dx) == 2:
                # Diagonal movement
                movement_cost *= np.sqrt(2)
            tentative_g_score = g_score[current_tuple] + movement_cost
            
            # Check if this path is better than any previous one
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current_tuple
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + manhattan_distance(neighbor, goal)
                
                # Add to open set if not already there
                if not any(neighbor == i[1] for i in open_set):
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
    
    # No path found
    return None

def manhattan_distance(a, b):
    """Calculate Manhattan distance between two points."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])