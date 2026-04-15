import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import chi2, norm
from pointpats import QStatistic

# Function to calculate and store (in list) all nn-d values and the corresponding point indices and the corresponding pairs indices
def unique_nearest_neighbor_distances(points):
    """
    Parameters:
        points: A 2D list of generated points (from the model)

    Returns:
        nn_distances : list of every unique nn-d values
        indices_pts : list of corresponding points indices of the nn-d
        recorded_pairs : list of tuple of point indices of the nn-d (d(i,j))
    """
    
    # Calculate all pairwise distances between each point
    distances = pdist(points)
    # Convert 'distances' into a square matrix
    dist_matrix = squareform(distances)
    
    # Initialize lists to store nn-d and corresponding point indices
    nn_distances = []
    indices_pts = []
    
    # Browse each point to find the nn-d value without duplicates
    num_points = len(points)
    recorded_pairs = set()
    
    for i in range(num_points):
        # Obtain distances from point i to other points
        distances_from_i = dist_matrix[i]
        # Ignore the distance of point i from itself (which is always 0)
        distances_from_i[i] = np.inf
        # Find the associated index of the nearest neighbor of point i
        j = np.argmin(distances_from_i)
        nearest_distance = distances_from_i[j]
        
        # Ensure that we only record the distance if (i, j) or (j, i) has not been recorded yet
        if (i, j) not in recorded_pairs and (j, i) not in recorded_pairs:
            nn_distances.append(nearest_distance)
            indices_pts.append(i)
            recorded_pairs.add((i, j))
    
    # Return the list of nn-d and the list of the corresponding point indices and pairs indices
    return nn_distances, indices_pts, recorded_pairs

# Function which is doing the Clark-Evans test and/or the Skellam one
def Clark_Evans_Skellam_function(lambda_,points,nn_distances,method):
    """
    Parameters:
        lambda_ : density of the model
        points: A 2D list of generated points (from the model)
        nn_distances : list of every unique nn-d values
        method : value to know which test to do

    Returns:
        S_m : Skellam's statistic
        critical_value : threshold value of Skellam' statistic
        alpha : significance level
        z_m : Chi-2 test value
        d_m : Mean of nn-d
    """
    
    # Skellam statistic test :
    if method == 0 or method == 2:
        # Occurrence of NN-Distance values
        nb_nn_d = len(nn_distances)
        
        # Calculate S_m (Skellam Statistic) :
        S_m = 0

        for i in range(nb_nn_d):
            S_m += nn_distances[i]**2
        S_m = 2 * lambda_ * np.pi * S_m

        # Test of the value under CSR Hypothesis
        alpha = 0.05  # Significance level
        df = 2 * nb_nn_d  # degree of freedom
        critical_value = chi2.ppf(1 - alpha, df)  # critical value of chi-2 (chi-square) distribution
        
        if method == 0:
            # Return Skellam value, the critical one and the level alpha
            return S_m, critical_value, alpha
    
    # Clark-Evans test :
    if method == 1 or method == 2:
        # Recover the nn-d of the points
        nn_distances, _, _ = unique_nearest_neighbor_distances(points)
        # Implement the number of nn-d 
        nb_nn_d = len(nn_distances) 
        # CLT Parameters for the normal distribution (mu : Expected Value, sigma : standard deviation/ sigma**2 : variance)
        mu = 1 / (2 * np.sqrt(lambda_))
        sigma = np.sqrt((4 - np.pi) / (len(points) * 4 * lambda_ * np.pi))

        # Part of Clark-Evans test
        
        # Calculate D_m
        d_m = 1 / nb_nn_d * np.sum(nn_distances)
        # D_m = 9.037

        # Calculate z_m
        z_m = (d_m - mu) / sigma

        # Return the z_m value to correspond to a normal distribution (N(0,1)), d_m value the mean of all nn-d
        return z_m, d_m
    
    # Return the outputs of both test
    return S_m, critical_value, alpha, z_m, d_m

# Calculating function of Z_m & D_m mean with a random number of points selected
def Clark_Evans_simulation(num_points,points,lambda_,N):
    """
    Parameters:
        num_points : Number of points of the model
        points: A 2D list of generated points (from the model)
        lambda_ : density of the model
        N : Number of simulation

    Returns:
        all_Z_m : List of z_m value
        all_D_m : List of d_m value
    """
    
    # Initialize the lists of z_m value and d_m value
    all_Z_m = []
    all_D_m = []
    
    # Simulate N times the algorithm to have N value of z_m and d_m
    for _ in range(N):
        # Random selection of 'half' of the points
        selected_indices = np.random.choice(num_points,2 * num_points // 3, replace=False)
        selected_points = points[selected_indices]
        
        # Calculation of nn-d
        nn_distances, _, _ = unique_nearest_neighbor_distances(selected_points)
        
        # Initialization of lists Z_m and D_m for this iteration
        Z_m = 0
        D_m = 0
        
        # Executing the Clark_Evans_Skellam_function
        Z_m, D_m = Clark_Evans_Skellam_function(lambda_, selected_points, nn_distances, method=1)
        
        # Storage of the new results done above
        all_Z_m.append(Z_m)
        all_D_m.append(D_m)
    
    # Return the lists of Z_m values and D_m values
    return all_Z_m, all_D_m
        
# Function of the behaviour of z_m, with the p-value method / two-tailed and 'conclusion on model type'
def p_value_behaviour(z_m,alpha,z_alpha_2):
    """
    Parameters:
        z_m : normal distribution value of our pattern
        alpha : significance level (in %)
        z_alpha_2 : significance level under normal distribution
    """
    
    # Reject CSR Hypothesis if |z_m| < z_alpha/2
    if np.abs(z_m) < z_alpha_2:
        print("\nDo not reject the CSR Hypothesis")
    else:
        print("\nReject the CSR Hypothesis")
        # Determination of the model in relation to z_m value
        if z_m < 0:
            print("Clustering model")
        else:
            print("Dispersion model")
        
    # Calculation of the p-value according to the value z_m
    # Clustering or random case
    if z_m < 0:
        p_value_cluster = norm.cdf(z_m)
        p_value_dispersion = np.inf
    # Dispersed or random case
    else:
        p_value_cluster = np.inf
        p_value_dispersion = 1 - norm.cdf(z_m)
    p_value_sides = 2 * norm.cdf(- np.abs(z_m))

    # Display the results of each p-value
    print(f"\np-value Cluster: {p_value_cluster}")
    print(f"p-value Dispersion: {p_value_dispersion}")
    print(f"p-value Two-Tails: {p_value_sides}")
    
    # Storage of the previous values
    p_value = [p_value_cluster,p_value_dispersion,p_value_sides]
    status = 0
    
    # Conclusion on the model according to the p-value
    # Dispersed or clustering model conclusion
    for i in range(0,len(p_value)):
        if p_value[i] < alpha:
            print("\nReject the CSR Hypothesis")
            if i == 0:
                print("Clustering Model")
            elif i == 1:
                print("Dispersion Model")
            else:
                print("Not a random model")
            status = 1

    # Random model conclusion
    if status == 0:
        print("\nDo not reject the CSR Hypothesis")
        print("Possible random model")
        
# Function to print the quadrat method
def quadrat_method(points):
    """
    Parameters:
        points : A 2D list of generated points (from the model)
    """
    
    # Quadrat analysis from our pattern
    qstat = QStatistic(points,nx=3,ny=3)
    qstat.plot()
    
    # Informations related to quadrat model
    # Chi-2 value
    print(f"\nChi-2 value quadrat (Chi-2): {qstat.chi2}")
    # Degree of freedom (number of boxes minus 1) (m - 1)
    print(f"Degrees of freedom (m-1): {qstat.df}")
    # Threshold to determine the type of model (Chi2 = (s**2 / n_mean) * (m - 1))
    print(f"s**2 / n_mean : {qstat.chi2/qstat.df}")
    # P-value of the quadrat method
    print(f"Chi-2 p-value : {qstat.chi2_pvalue}\n")
    
    # Conclusion related to the previous results
    # Dispersed case
    if qstat.chi2/qstat.df < 1:
        # Threshold to affirm that the model is dispersed
        if qstat.chi2/qstat.df < 0.8:
            print("The pattern is certainly dispersed")
        # Possibility to be random or dispersed
        else:
            print("The pattern is possibly dispersed or random")
    # Random case
    elif qstat.chi2/qstat.df == 1:
        print("The pattern is built randomly")
    # Clustered case
    else:
        # Threshold to affirm that the model is clustered
        if qstat.chi2/qstat.df < 6:
            print("The pattern is possibly clustered or random")
        # Possibility to random or clustered
        else:
            print("The pattern is certainly clustered")
            
# Function to store the distance where the observation value is above the simulations
def get_significant_distances(support, statistic, simulations):
    """
    Parameters:
        support : 1D array of distances
        statistic : 1D array of observed statistics
        simulations : 2D array of simulated statistics (each row corresponds to a different simulation)

    Returns:
        significant_distances : list of distances where the observed statistic is strictly greater than all simulated statistics
    """
    significant_distances = []
    for i, dist in enumerate(support):
        if statistic[i] > np.max(simulations[:, i]):
            significant_distances.append(dist)
    return significant_distances