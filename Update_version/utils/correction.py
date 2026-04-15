import numpy as np
import math

# Function to calculate distance between all pairs points model
def calculate_distances(points):
    """
    Parameters:
        points : A 2D list of generated points (from the model)

    Returns:
        distances : Matrix which store the distance between each pair points
    """
    
    # Length of the list of points
    n = len(points)
    # Matrix to stored the distance values between point i and point j
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            # Calculation of the Euclidean distance between i and j
            distances[i, j] = np.linalg.norm(points[i] - points[j])
    # Return the matrix of distances
    return distances

# Function used to calculate the value e1 and e2 important for the Ripley's correction
def calculate_e1_e2(point, region):
    """
    Parameters:
        point : Coordinates of a model point
        region : list of for value of each side of the area (square or rectangle)

    Returns:
        e1 : minimum value between our point and one side of the region
        e2 : second minimum value between our point and another side of the region
    """
    
    # Recovery of the x-axis in x and y-axis in y which represent the coordinates of the point
    x, y = point
    
    # Limits coordinates of our region
    x_min, x_max, y_min, y_max = region
    
    # Sampling of the minimum distance between the point and a side
    e1 = min(x - x_min, x_max - x, y - y_min, y_max - y)
    # Distances between the point and each side of the region
    distances = [x - x_min, x_max - x, y - y_min, y_max - y]
    # REmoving the minimum value already stored
    distances.remove(e1)
    # Sampling of the minimum distance between the point and a side
    e2 = min(distances)
    
    # Return the two minimum distances between a side and the studied point
    return e1, e2

# Function to calculate the weight/influence of every point in relation to the edge effects problems (Ripley's Correction)
def calculate_weight_Ripley(point, region, h):
    """
    Parameters:
        point : Coordinates of a model point
        region : list of for value of each side of the area (square or rectangle)
        h : distance of the radius of the K-function
        
    Return:
        wij : Value of the Ripley's weight of the selected point"""

    # Calculation of the two minimum distances between a side of the region and the selected point
    e1, e2 = calculate_e1_e2(point, region)
    
    # Calculation of the Proportion of Circle Circumference of the point regarding the following cases
    # Intersection with a single border
    if (e1 < h and e2 >= h) or (e1 >= h and e2 < h):  # Case A
        w_ij = 1 - (np.arccos(e1/h) / np.pi) if e1 < h else 1 - (np.arccos(e2/h) / np.pi)
    elif e1 < h and e2 < h:
        # Intersection with two borders, excluding the corner
        if e1**2 + e2**2 >= h**2:  # Case B
            w_ij = 1 - (np.arccos(e1/h) + np.arccos(e2/h)) / np.pi
        # Intersection with two borders, including the corner
        else:  # Case C
            w_ij = 3/4 - (np.arccos(e1/h) + np.arccos(e2/h)) / (2 * np.pi)
    # No edge effects applied on the point
    else:   # Case D
        w_ij = 1.0  # No intersection with border
        
    # The weight of our point is the ratio between circle's circumference inside the area on the total circle's circumference
    return w_ij

# Function to calculate the weight/influence of every point in relation to the edge effects problems (Besag's Correction)
def calculate_weight_Besag(point, region, h):
    """
    Parameters:
        point : Coordinates of a model point
        region : list of for value of each side of the area (square or rectangle)
        h : distance of the radius of the K-function
        
    Return:
        wij : Value of the Besag's weight of the selected point"""

    # Calculation of the two minimum distances between a side of the region and the selected point
    e1, e2 = calculate_e1_e2(point, region)
    
    # Calculation of the Proportion of Circle Circumference of the point regarding the following cases
    # Intersection with a single border
    if (e1 < h and e2 >= h) or (e1 >= h and e2 < h):  # Case A
        if e1 < h:
            w_ij = 1 - (np.arccos(e1/h) / np.pi) + e1 * np.sqrt(h**2 - e1**2) / (np.pi * h**2)
        else:
            w_ij = 1 - (np.arccos(e2/h) / np.pi) + e2 * np.sqrt(h**2 - e2**2) / (np.pi * h**2)
    elif e1 < h and e2 < h:
        # Intersection with two borders, excluding the corner
        if e1**2 + e2**2 >= h**2:  # Case B
            w_ij = 1 - (np.arccos(e1/h) + np.arccos(e2/h)) / np.pi + (e1 * np.sqrt(h**2 - e1**2) + e2 * np.sqrt(h**2 - e2**2)) / (np.pi * h**2)
        # Intersection with two borders, including the corner
        else:  # Case C
            w_ij = 3/4 - (np.arccos(e1/h) + np.arccos(e2/h)) / (2 * np.pi) + (2 * e1 * e2 + e1 * np.sqrt(h**2 - e1**2) + e2 * np.sqrt(h**2 - e2**2)) / (2 * np.pi * h**2)
    # No edge effects applied on the point
    else:   # Case D
        w_ij = 1.0  # No intersection with border
        
    # The weight of our point is the ratio between circle's circumference inside the area on the total circle's circumference
    return w_ij

# Function to calculate the weight of each point using Wiegand and Moloney correction (Besag's numeric approach)
def calculate_weight_Wiegand_Moloney(point, img_array_area, h):
    """
    Parameters:
        point : coordinates of the point (event)
        img_array_area : Binary matrix of the area (1 represents the area, 0 represents outside)
        h : Distance of the radius of the K-function

    Returns:
        weight : weight for our point
    """
    # Get the whole dimensions of the image
    n_rows, n_cols = img_array_area.shape
    
    # Calculate the number of pixels in a circle with radius h
    # Coordinates of the regular area which contain the circle
    y, x = np.ogrid[-h:h+1, -h:h+1]
    # Generate a binary matrix to know the total number of pixels which represent the circle
    mask = x**2 + y**2 <= h**2
    # Sum all pixels of the subregion which has 1 in value
    num_pixels_in_circle = np.sum(mask)
    
    # Coordinates of our point
    x_center, y_center = point
    # Extract the sub-array around the point of side with minimum 2h (make sure that the following boundary values are int) 
    x_min = int(max(x_center - h, 0))
    x_max = min(x_center + h, n_cols)
    y_min = int(max(y_center - h, 0))
    y_max = min(y_center + h, n_rows)
    
    # Conditions to be sure than all the circle is taking into account for the weight ratio
    if x_max > int(np.round(x_max)):
        x_max = int(x_max) + 1
    else:
        x_max = int(x_max)
    if y_max > int(np.round(y_max)):
        y_max = int(y_max) + 1
    else:
        y_max = int(y_max)
    
    # Extract a sub area which contains the circle
    sub_array = img_array_area[y_min:y_max, x_min:x_max]
    
    # Be sure to have the radius value as an integer to pick up a correct value of bondary index
    if h > int(h):
        h = int(h)+1
    else:
        h = int(h)
    
    # Calculate the number of pixels in the circle within the area
    num_pixels_in_area_circle = np.sum(sub_array[mask[max(y_min-y_center+h,0):y_max-y_center+h, max(x_min-x_center+h,0):x_max-x_center+h]])
    # Calculate the weight
    weight = num_pixels_in_area_circle / num_pixels_in_circle
    
    # Return the weight value following the method
    return weight

# Function to calculate the weight of every model points with the application of the Ripley's Correction principe (take in count the edge effects)
def compute_weights(points, h, region, img_array_area, method):
    """
    Calculates the weights w_ij for each pair of points in the given region.

    Parameters:
    points : np.ndarray
        2D table of point coordinates.
    h : float
        Value of the K-function radius.
    region : list
        List [x_min, x_max, y_min, y_max] defining region boundaries.
    method : int
        Index of the correction method selected

    Returns:
    weights : np.value(n,n)
        Matrix of weights values of points i and j (wij <= 1).
    """
    
    # Initialisation of the matrix of weights with the basic case value (1)
    weights = np.ones((len(points),len(points)))
    
    # Determination of the weight w_ij which represents the ration of the circle circumference in the area on the total circle circumference
    # The center of the circle is point i with the associated radius d(i,j) 
    for i in range(len(points)):
        for j in range(len(points)):
            # Useless to calculate the weight of the circle with a radius d(i,i) = 0
            if i != j:
                # Coordinates of the point i
                s_i = points[i]
                # Coordinates of the point j
                s_j = points[j]
                # Weight of the point i related to the point j
                w_ij = compute_w_ij(s_i, s_j, h, region, img_array_area, method)
                # We're switching the initial weight's value with the real new one
                weights[i, j] = w_ij
    
    # Return the matrix of weights updated
    return weights

# Function to calculate the weight of the point s_i related to a point s_j
def compute_w_ij(s_i, s_j, h, region, img_array_area, method):
    """
    Calculates the weight w_ij for two points s_i and s_j,
    based on their distance and K-function's radius h.

    Parameters:
    s_i : list
        Coordinates of point s_i [x, y].
    s_j : list
        Coordinates of point s_j [x, y].
    h : float
        Radius h to define the region of interest (of K-function).
    region : list
        Defines the region boundaries [x_min, x_max, y_min, y_max].
    method : int
        Index of the correction method selected

    Returns:
    float
        Weight w_ij.
    """
    x_i, y_i = s_i
    x_j, y_j = s_j
    
    # Calculate the distance between points s_i and s_j
    distance_ij = math.sqrt((x_j - x_i)**2 + (y_j - y_i)**2)
    
    # If distance is greater than h, weight is 1.0 (Null)
    if distance_ij > h:
        return 1.0
    else:
        # Calculate the radius of the circle connecting s_i and s_j
        radius = distance_ij
        
        # Calculate the weight of the points pattern due to edge effects with a selected method
        if method == 0:
            # Calculate the part of the circle's circumference inside the region (Ripley's correction)
            w_ij = calculate_weight_Ripley(s_i,region,radius)
        elif method == 1:
            # Calculate the part of the circle's area inside the region (Besag's correction)
            w_ij = calculate_weight_Besag(s_i, region,radius)
        elif method == 2:
            w_ij = calculate_weight_Wiegand_Moloney(s_i, img_array_area, radius)
        else:
            # Wrong index of method, no method applied
            print("No Correction method selected")
            w_ij = 1.0
        
        # Return the weight of each point related to edge effects
        return w_ij

# Function to calculate the K-value with known fixed h with the edge effects corrections
def Correction_K_function(points, region, h, img_array_area, method, hull=None):
    """
    Parameters:
        points : A 2D list of generated points (from the model)
        region : list of for value of each side of the area (square or rectangle)
        h : distance of the radius of the K-function
        method : Index of the correction method selected
    
    Returns:
        K_corr : Value of the corrected K-function (with Ripley's correction)
        K_norm : Value of the uncorrected K-function (without Ripley's correction)
        weights : Matrix of weights of the point i (center) and j (arc)
        w : Matrix of the weight of point i and j without edge effects (d(i,j) = 1)"""
    
    # Length of the list of points
    n = len(points)
    
    # Calculation of the distances of our model points
    distances = calculate_distances(points)
    
    # Implementation of all weights related to Ripley's correction or another selected method (weights) and without it (w)
    w = np.ones((len(points), len(points)))  
    weights = compute_weights(points, h, region, img_array_area, method)
    
    # Create a mask for distances less than or equal to h and greater than 0 to exclude self-distances
    mask = (distances <= h) & (distances > 0)
    
    # Calculate K_corrected and K_normal based on the mask and weights
    K_corrected = np.sum(mask / weights)
    K_normal = np.sum(mask)
    
    # Possibility to calculate the area of the events for K-function values
    if hull is not None:
        # Calculate K values
        K_corrected = (hull.area / n**2) * K_corrected
        K_normal = (hull.area / n**2) * K_normal
    else:
        # Regular area value
        area_R = (region[1] - region[0]) * (region[3] - region[2])
        # Calculate K values
        K_corrected = (area_R / n**2) * K_corrected
        K_normal = (area_R / n**2) * K_normal
    
    # Return Corrected and Uncorrected values of K-function and the associated weight's list of each point
    return K_corrected, K_normal, weights, w

# Calculation of the K-function values under a corrected condition
def weighted_ripley_k(points, max_dist, region, img_array_area, method, hull_pattern, support=40):
    """
    Calculate the weighted Ripley's K function for a set of points with weights.
    
    Parameters:
        points : A 2D numpy array of points.
        max_dist : The maximum distance to calculate K(d).
        region : list of for value of each side of the area (square or rectangle)
        img_array_area : Binary matrix to represent the data area
        method : Index of the selected correction method
        support : Number of distances to evaluate.
    
    Returns:
        distances : the evaluated distances h divide by support value, 
        K_values : the corresponding K(d) values with the correction method specify.
        K_norm : the corrsponding K(d) values without a correction method applied.
    """
    
    # Division of the distances between 0 and the max kNN value by support
    distances = np.linspace(0, max_dist, support)
    # Initialisation of the K_{RC}(d) values
    K_values = np.zeros(support)
    K_norm = np.zeros(support)
    
    # Calculation of the the K-function RC value following the previous formula :
    for i, h in enumerate(distances):
        # Update of the weight's matrix according the h K-function radius
        K_corr, K_wout, _, _ = Correction_K_function(points, region, h, img_array_area, method, hull_pattern)
        # Sub the initialized values by the one from the formula
        K_values[i] = K_corr
        K_norm[i] = K_wout
    # Return the distances where K-values has been calculated
    return distances, K_values, K_norm

# Calculation of the L-function values under a corrected condition
def weighted_ripley_l(K_values, K_norm):
    """
    Parameters:
        K_values : Values of the weighted K-function
        K_norm : Values of the normal K-function
        
    Return:
        L_values : Values of the weighted L-function
        L_norm : Values of the normal L-function
    """
    
    # L-function values related to the L formula
    L_values = np.sqrt(K_values / np.pi)
    L_norm = np.sqrt(K_norm / np.pi)
    
    return L_values, L_norm
