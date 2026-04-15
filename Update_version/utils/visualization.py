import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import pointpats
from pointpats import distance_statistics

from .spatial_analysis import get_significant_distances
from .statistical_analysis import check_normality_and_display_first
from .correction import weighted_ripley_k, weighted_ripley_l

# Function to plot the kernel density behaviour related to our model
def plot_kde_with_weights(points, weights, region, method):
    # Store all points in coordinates variables
    x, y = points.T
    # Store the boundary value of the area
    x_min, x_max, y_min, y_max = region
    # Generate the grid for the kernel density generation
    xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    
    # Apply the kernel density on our point pattern
    kde = gaussian_kde(points.T, weights=np.sum(weights, axis=1))
    f = np.reshape(kde(positions).T, xx.shape)
    
    # Gather all the weight on each point
    weight_list = np.zeros(len(points))
    for i in range(len(points)):
        weight_list[i] = np.sum(weights[i,:])
    
    # Plot the KDE  
    plt.figure(figsize=(9, 9))
    plt.contourf(xx, yy, f, cmap='viridis_r', alpha=0.55)
    plt.colorbar(label='Level of the KDE')
    plt.scatter(x, y, c=weight_list, edgecolor='black')
    
    # Select the method correction
    if len(np.unique(weight_list,return_counts=False)) == 1:
        plt.title('KDE without Ripley\'s Correction')
    elif method == 0:
        plt.title('KDE with Ripley\'s Correction Weights')
    elif method == 1:
        plt.title('KDE with Besag\'s Correction Weights')
    elif method == 2:
        plt.title('KDE with WM\'s Correction Weights')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
    
# Function to display the correction method selected impact and the weight of all model points
def display_kernel_weights(points, weights, region, radius):
    """
    Parameters:
        points : A 2D list of generated points (from the model)
        weights : List of the weight of each point
        region : list of for value of each side of the area (square or rectangle)
        radius : distance of the radius of the K-function
    """

    plt.figure(figsize=(10, 8))
    
    # Sum all the weights for each point
    weight_list = np.sum(weights, axis=1)
    
    # Plot the model
    scatter = plt.scatter(points[:, 0], points[:, 1], c=weight_list, cmap='coolwarm', edgecolor='black', label='Point')
    plt.colorbar(scatter, label='Weight')
    
    # Plot the limits of the region/area for a regular one
    plt.axhline(region[2] + radius, linestyle='--', c='yellow')
    plt.axhline(region[3] - radius, linestyle='--', c='yellow')
    plt.axvline(region[0] + radius, linestyle='--', c='yellow')
    plt.axvline(region[1] - radius, linestyle='--', c='yellow')

    # Set title of the graph
    plt.title('Weight value of points model in relation to K-function radius h')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()
    
# Function to display the correction method selected impact and the weight of all model points
def display_kernel_weights_bis(points, weights, outline):
    """
    Parameters:
        points : A 2D list of generated points (from the model)
        weights : List of the weight of each point
        outline : circumrference coordinates of the irregular area
    """

    plt.figure(figsize=(10, 8))
    
    # Sum all the weights for each point
    weight_list = np.sum(weights, axis=1)
    
    # Plot the model
    scatter = plt.scatter(points[:, 0], points[:, 1], c=weight_list, cmap='coolwarm', edgecolor='black', label='Point')
    plt.colorbar(scatter, label='Weight')

    # Plot the contour within the region of interest for irregular one
    adjusted_outline = [np.array([[y, x] for x, y in contour]) for contour in outline]
    for a_outline in adjusted_outline:
        plt.plot(a_outline[:, 0], a_outline[:, 1], linewidth=2, color='blue', label='Contours')

    # Set title of the graph
    plt.title('Weight value of points model in relation to K-function radius h')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()
    
# Display function of Spatial points pattern and the associated NN-Distances
def display_spatial_points_analysis(coordonnees, area_dim, nb_points, ind_nn_d, nn_distances, tuple_dist):
    """
    Parameters:
        coordonnees : A 2D list of generated points (from the model)
        area_dim : Dimension of the area (assumed to be square)
        nb_points : Number of points of the model
        ind_nn_d : list of corresponding points indices of the nn-d
        nn_distances : list of every unique nn-d values
        tuple_dist : list of every couple which represent the unique nn-d values
    """

    # Visualization of the pattern following the Poisson Approximation
    plt.figure(figsize=(8, 8))
    plt.scatter(coordonnees[:, 0], coordonnees[:, 1], c='blue', marker='o', s=10, label='Points')

    # Map out the circles to visualize the NN-Distances stored and calculated before
    for i in range(len(ind_nn_d)):  # You can add a third value 'k' to display 1 circle on k in the graph
        # Here, we print only the circle of the points registered in the list of indices of nn-d 'ind_nn_d'
        circle = plt.Circle(coordonnees[ind_nn_d[i]], nn_distances[i], color='red', fill=False, linestyle='dotted')
        # Add the display of the circle number i to the others done before
        plt.gca().add_patch(circle)
        
    # Plot the lines representing the NN-Distances
    for (i, j) in tuple_dist:
        pt1 = coordonnees[i]
        pt2 = coordonnees[j]
        plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], '--', color="lightgreen")

    # Legend for NN-D circles
    legend_circle = Patch(color='red', linestyle='dotted', fill=False, label='Circles of NN-D')
    # Legend for the pattern points
    legend_point = Line2D([0], [0], color='blue', marker='o', linestyle='', label='Pattern Points')
    # Legend for the NN-D lines
    legend_line = Line2D([0], [0], color='lightgreen', linestyle='--', label='Lines of NN-D')

    # Commands to print the Spatial Points Model
    plt.title('Spatial Poisson Process (under CSR or forced) and kNN (K-Nearest Neighbors)')
    plt.xlabel('Coordinate X')
    plt.ylabel('Coordinate Y')
    plt.legend(handles=[legend_point, legend_circle, legend_line])
    # plt.grid(True)
    plt.show()
    
# Display function of Ripley's G function (if red line above the blue one -> clustering, under dispersion)
def display_Ripley_G(points, max_dist, data_info, save_path, hull_pattern=None):
    """
    Parameters:
        points : A 2D list of generated points (from the model)
        max_dist : maximum distance of the nn-d
        data_info : additional information to specify the data (e.g., 'data1', 'data2', etc.)
        hull_pattern: bounding box, scipy.spatial.ConvexHull, shapely.geometry.Polygon the hull used to construct a random sample pattern 
        save_path : path where the figure should be saved
    """
    
    # Ensure the save directory exists
    full_save_path = os.path.join(save_path, data_info)
    os.makedirs(full_save_path, exist_ok=True)
    
    # Calculation of the Ripley's G function
    g_test = distance_statistics.g_test(points, support=40, hull=hull_pattern, keep_simulations=True, n_simulations=1000)
    
    # Get significant distances where the observed G function is greater than all simulations
    significant_distances = get_significant_distances(g_test.support, g_test.statistic, g_test.simulations)
    
    # Print significant distances
    print("Significant distances where observed G function is greater than all simulations:")
    print(significant_distances)
    
    # Parameters for the display
    f, ax = plt.subplots(1, 2, figsize=(9, 3), gridspec_kw=dict(width_ratios=(6, 3)))
    
    # Plot all the simulations with very fine lines (black lines)
    ax[0].plot(g_test.support, g_test.simulations.T, color="k", alpha=0.01)
    
    # and show the average of simulations (blue line)
    ax[0].plot(
        g_test.support,
        np.median(g_test.simulations, axis=0),
        color="cyan",
        label="median simulation",
    )

    # and the observed pattern's G function (red line)
    ax[0].plot(g_test.support, g_test.statistic, label="observed", color="red")
    
    IC_distance_first = check_normality_and_display_first(g_test.support, g_test.simulations, g_test.statistic, data_info, save_path)
    IC_distance_second = check_normality_and_display_first(g_test.support, g_test.simulations, g_test.statistic, data_info, save_path, method=2)
    
    if IC_distance_first:
        ax[0].axvline(IC_distance_first, color='g', linestyle='dashed', linewidth=1, label='distance switch state')
    if IC_distance_second:
        ax[0].axvline(IC_distance_second, color='g', linestyle='dashed', linewidth=1)
    

    # clean up labels and axes
    ax[0].set_xlabel("distance")
    ax[0].set_ylabel("% of nearest neighbor\ndistances shorter")
    ax[0].legend()
    # max_dist represent the maximum limit where it means to calculate the Ripley's function (max_dist = largest kNN)
    ax[0].set_xlim(0, max_dist)
    ax[0].set_title(r"Ripley's $G(d)$ function")

    # plot the pattern itself on the right frame
    ax[1].scatter(*points.T)

    # and clean up labels and axes there, too
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[1].set_xticklabels([])
    ax[1].set_yticklabels([])
    ax[1].set_title("Pattern")
    f.tight_layout()
    
    # Save the figure
    file_path = os.path.join(full_save_path, 'G_function.png')
    plt.savefig(file_path)
    plt.show()
    
    check_normality_and_display_first(g_test.support, g_test.simulations, g_test.statistic, data_info, save_path, method=1, display='on')
    check_normality_and_display_first(g_test.support, g_test.simulations, g_test.statistic, data_info, save_path, method=2, display='on')
    
    # Return the list of the significant which figure out a possible aggregated pattern
    return significant_distances
    
    
# Display function of Ripley's F function (if red line under the blue one -> clustering, above dispersion)
def display_Ripley_F(points, max_dist, data_info, save_path, hull_pattern=None):
    """
    Parameters:
        points : A 2D list of generated points (from the model)
        max_dist : maximum distance of the nn-d
        data_info : additional information to specify the data (e.g., 'data1', 'data2', etc.)
        hull_pattern: bounding box, scipy.spatial.ConvexHull, shapely.geometry.Polygon the hull used to construct a random sample pattern 
        save_path : path where the figure should be saved
    """
    
    # Ensure the save directory exists
    full_save_path = os.path.join(save_path, data_info)
    os.makedirs(full_save_path, exist_ok=True)
    
    # Calculation of the Ripley's F function
    f_test = distance_statistics.f_test(points, support=40, hull=hull_pattern, keep_simulations=True, n_simulations=1000)
    
    # Get significant distances where the observed G function is greater than all simulations
    significant_distances = get_significant_distances(f_test.support, f_test.statistic, f_test.simulations)
    
    # Print significant distances
    print("Significant distances where observed F function is greater than all simulations:")
    print(significant_distances)
    
    # Parameters for the display
    f, ax = plt.subplots(1, 2, figsize=(9, 3), gridspec_kw=dict(width_ratios=(6, 3)))

    # plot all the simulations with very fine lines (black lines)
    ax[0].plot(f_test.support, f_test.simulations.T, color="k", alpha=0.01)
    
    # and show the average of simulations (blue line)
    ax[0].plot(
        f_test.support,
        np.median(f_test.simulations, axis=0),
        color="cyan",
        label="median simulation",
    )


    # and the observed pattern's F function (red line)
    ax[0].plot(f_test.support, f_test.statistic, label="observed", color="red")

    # clean up labels and axes
    ax[0].set_xlabel("distance")
    ax[0].set_ylabel("% of nearest point in pattern\ndistances shorter")
    ax[0].legend()
    # max_dist represent the maximum limit where it means to calculate the Ripley's function (max_dist = largest kNN)
    ax[0].set_xlim(0, max_dist)
    ax[0].set_title(r"Ripley's $F(d)$ function")

    # plot the pattern itself on the right frame
    ax[1].scatter(*points.T)

    # and clean up labels and axes there, too
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[1].set_xticklabels([])
    ax[1].set_yticklabels([])
    ax[1].set_title("Pattern")
    f.tight_layout()
    
    # Save the figure
    file_path = os.path.join(full_save_path, 'F_function.png')
    plt.savefig(file_path)
    plt.show()
    
# Display function of Ripley's K function (if red line above the blue one -> clustering, under dispersion)
def display_Ripley_K(points, max_dist, data_info, save_path, hull_pattern=None):
    """
    Parameters:
        points : A 2D list of generated points (from the model)
        max_dist : maximum distance of the nn-d
        data_info : additional information to specify the data (e.g., 'data1', 'data2', etc.)
        hull_pattern: bounding box, scipy.spatial.ConvexHull, shapely.geometry.Polygon the hull used to construct a random sample pattern 
        save_path : path where the figure should be saved
    """
    
    # Ensure the save directory exists
    full_save_path = os.path.join(save_path, data_info)
    os.makedirs(full_save_path, exist_ok=True)
    
    # Calculation of the Ripley's K function
    k_test = distance_statistics.k_test(points, support=40, hull=hull_pattern, keep_simulations=True, n_simualtions=1000)
    
    # Get significant distances where the observed G function is greater than all simulations
    significant_distances = get_significant_distances(k_test.support, k_test.statistic, k_test.simulations)
    
    # Print significant distances
    print("Significant distances where observed K function is greater than all simulations:")
    print(significant_distances)

    # Parameters for the display
    fig, ax = plt.subplots(1, 2, figsize=(9, 3), gridspec_kw=dict(width_ratios=(6, 3)))

    # plot all the simulations with very fine lines (black lines)
    ax[0].plot(k_test.support, k_test.simulations.T, color="k", alpha=0.01)
    
    # and show the average of simulations (blue line)
    ax[0].plot(
        k_test.support,
        np.median(k_test.simulations, axis=0),
        color="cyan",
        label="median simulation",
    )

    # and the observed pattern's K function (red line)
    ax[0].plot(k_test.support, k_test.statistic, label="observed", color="red")

    # clean up labels and axes
    ax[0].set_xlabel("Distance")
    ax[0].set_ylabel("K(d) value")
    ax[0].legend()
    # max_dist represent the maximum limit where it means to calculate the Ripley's function (max_dist = largest kNN)
    ax[0].set_xlim(0, max_dist)
    ax[0].set_title(r"Ripley's $K(d)$ function")

    # plot the pattern itself on the right frame
    ax[1].scatter(*points.T)

    # and clean up labels and axes there, too
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[1].set_xticklabels([])
    ax[1].set_yticklabels([])
    ax[1].set_title("Pattern")
    
    fig.tight_layout()
    
    # Save the figure
    file_path = os.path.join(full_save_path, 'K_function.png')
    plt.savefig(file_path)
    plt.show()
    
# Display function of Ripley's L function (if red line above the blue one -> clustering, under dispersion)
def display_Ripley_L(points, max_dist, data_info, save_path, hull_pattern=None):
    """
    Parameters:
        points : A 2D list of generated points (from the model)
        max_dist : maximum distance of the nn-d
        data_info : additional information to specify the data (e.g., 'data1', 'data2', etc.)
        hull_pattern: bounding box, scipy.spatial.ConvexHull, shapely.geometry.Polygon the hull used to construct a random sample pattern 
        save_path : path where the figure should be saved
    """
    
    # Ensure the save directory exists
    full_save_path = os.path.join(save_path, data_info)
    os.makedirs(full_save_path, exist_ok=True)
    
    # Calculation of the Ripley's L function
    l_test = distance_statistics.l_test(points, support=40, hull=hull_pattern, keep_simulations=True, n_simualtions=1000)
    
    # Get significant distances where the observed G function is greater than all simulations
    significant_distances = get_significant_distances(l_test.support, l_test.statistic, l_test.simulations)
    
    # Print significant distances
    print("Significant distances where observed L function is greater than all simulations:")
    print(significant_distances)

    # Parameters for the display
    fig, ax = plt.subplots(1, 2, figsize=(9, 3), gridspec_kw=dict(width_ratios=(6, 3)))

    # plot all the simulations with very fine lines (black lines)
    ax[0].plot(l_test.support, l_test.simulations.T, color="k", alpha=0.01)
    
    # and show the average of simulations (blue line)
    ax[0].plot(
        l_test.support,
        np.median(l_test.simulations, axis=0),
        color="cyan",
        label="median simulation",
    )

    # and the observed pattern's K function (red line)
    ax[0].plot(l_test.support, l_test.statistic, label="observed", color="red")

    # clean up labels and axes
    ax[0].set_xlabel("Distance")
    ax[0].set_ylabel("L(d) value")
    ax[0].legend()
    # max_dist represent the maximum limit where it means to calculate the Ripley's function (max_dist = largest kNN)
    ax[0].set_xlim(0, max_dist)
    ax[0].set_title(r"Ripley's $L(d)$ function")

    # plot the pattern itself on the right frame
    ax[1].scatter(*points.T)

    # and clean up labels and axes there, too
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[1].set_xticklabels([])
    ax[1].set_yticklabels([])
    ax[1].set_title("Pattern")
    
    fig.tight_layout()
    
    # Save the figure
    file_path = os.path.join(full_save_path, 'L_function.png')
    plt.savefig(file_path)
    plt.show()
    
# Display function of Ripley's J function (if the line above the threshold equal to 1 -> dispersion, otherwise clustering)
# J function works as a ratio (J(d) = 1, random pattern)
def display_Ripley_J(points, max_dist, data_info, save_path, hull_pattern=None):
    """
    Parameters:
        points : A 2D list of generated points (from the model)
        max_dist : maximum distance of the nn-d
        data_info : additional information to specify the data (e.g., 'data1', 'data2', etc.)
        hull_pattern: bounding box, scipy.spatial.ConvexHull, shapely.geometry.Polygon the hull used to construct a random sample pattern 
        save_path : path where the figure should be saved
    """
    
    # Ensure the save directory exists
    full_save_path = os.path.join(save_path, data_info)
    os.makedirs(full_save_path, exist_ok=True)
    
    # Calculation of the Ripley's J function
    j_test = distance_statistics.j_test(points, support=40, hull=hull_pattern)
    
    # Get significant distances where the observed G function is greater than all simulations
    significant_distances = get_significant_distances(j_test.support, j_test.statistic, j_test.simulations)
    
    # Print significant distances
    print("Significant distances where observed J function is greater than all simulations:")
    print(significant_distances)

    # Parameters for the display
    fig, ax = plt.subplots(1, 2, figsize=(9, 3), gridspec_kw=dict(width_ratios=(6, 3)))

    # and the observed pattern's J function (red line)
    ax[0].plot(j_test.support, j_test.statistic, label="observed", color="orange")
    
    ax[0].axhline(1, linestyle=':',color='k')

    # clean up labels and axes
    ax[0].set_xlabel("Distance")
    ax[0].set_ylabel("J(d) value")
    ax[0].legend()
    # max_dist represent the maximum limit where it means to calculate the Ripley's function (max_dist = largest kNN)
    ax[0].set_xlim(0, max_dist)
    ax[0].set_title(r"Ripley's $J(d)$ function")

    # plot the pattern itself on the right frame
    ax[1].scatter(*points.T)

    # and clean up labels and axes there, too
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[1].set_xticklabels([])
    ax[1].set_yticklabels([])
    ax[1].set_title("Pattern")
    
    fig.tight_layout()

    # Save the figure
    file_path = os.path.join(full_save_path, 'J_function.png')
    plt.savefig(file_path)
    plt.show()
            
# Function to display all Ripley's functions (F,G,J,K,L)
def Ripley_dispaly_all(points,max_dist,data_info,hull=None):
    """
    Parameters:
        points : A 2D list of generated points (from the model)
        max_dist : MAX distance of the nn-d list for the printed interval of Ripley's functions
        data_info : additional information to specify the data (e.g., 'data1', 'data2', etc.)
        hull: bounding box, scipy.spatial.ConvexHull, shapely.geometry.Polygon the hull used to construct a random sample pattern  
    """
    
    # K-function (F,G,K,L,J) analysis and display
    print("\nDisplay of Ripley's functions :")
    _ = display_Ripley_G(points,max_dist,data_info,hull_pattern=hull)
    display_Ripley_F(points,max_dist,data_info,hull_pattern=hull)
    display_Ripley_K(points,max_dist,data_info,hull_pattern=hull)
    display_Ripley_L(points,max_dist,data_info,hull_pattern=hull)
    display_Ripley_J(points,max_dist,data_info,hull_pattern=hull)
    
# Function to display the K Ripley's function as the function 'display_Ripley_K' by adding the plot of the observed K-values under RC
def display_Ripley_K_L_weight(points, weights, unweight, max_dist, region, method, img_array_area, data_info, save_path, hull_pattern=None):
    """
    Parameters:
        points : A 2D numpy array of generated points (from the model)
        weights : A 2D numpy array of weights with w[i,j] the weight of the point i related to the point j
        unweight : A 2D numpy array of weights (not applied) with w[i,j] = 1 value
        max_dist : maximum distance for Ripley's K function
        region : list of for value of each side of the area (square or rectangle)
        method : Index of correction method selected
        img_array_area : Binary matrix to represent the data area
        data_info : additional information to specify the data (e.g., 'data1', 'data2', etc.)
        hull_pattern: bounding box, scipy.spatial.ConvexHull, shapely.geometry.Polygon the hull used to construct a random sample pattern 
        save_path : path where the figure should be saved
    """
    
    # Ensure the save directory exists
    full_save_path = os.path.join(save_path, data_info)
    os.makedirs(full_save_path, exist_ok=True)

    # Calculate the weighted and unweighted Ripley K function
    distances, K_values, K_norm = weighted_ripley_k(points, max_dist, region, img_array_area, method, hull_pattern)

    # Calculate the Ripley K function furnished by 'pointpats' library
    K_test = pointpats.k_test(points, support=40, hull=hull_pattern, keep_simulations=True, n_simulations=1000)
    
    # Get significant distances where the observed G function is greater than all simulations
    significant_distances = get_significant_distances(K_test.support, K_test.statistic, K_test.simulations)
    
    # Print significant distances
    print("Significant distances where observed K function is greater than all simulations:")
    print(significant_distances)

    # Parameters for the display
    fig, ax = plt.subplots(1, 3, figsize=(14, 4), gridspec_kw=dict(width_ratios=(4 ,6, 4)))
    
    # Gather all the weight value for each event
    unweight_list = np.zeros(len(points))
    for i in range(len(points)):
        unweight_list[i] = np.sum(unweight[i,:])
    
    # Plot the pattern itself on the left frame
    scatter = ax[0].scatter(points[:, 0], points[:, 1], c=unweight_list, cmap='viridis')
    ax[0].set_title("Pattern")
    plt.colorbar(scatter, ax=ax[0], label='No Weights')

    # Clean up labels and axes there, too 
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_xticklabels([])
    ax[0].set_yticklabels([])
    
    
    # Plot all the simulations with very fine lines (black lines)
    ax[1].plot(K_test.support, K_test.simulations.T, color="k", alpha=0.01)
    
    # And show the average of simulations (blue line)
    ax[1].plot(
        K_test.support,
        np.median(K_test.simulations, axis=0),
        color="cyan",
        label="median simulation",
    )
    
    # And the observed pattern's K function (red line)
    ax[1].plot(K_test.support, K_test.statistic, label="observed", color="red")
    
    # Comparison with the analytic value (green line)
    ax[1].plot(distances, K_norm, label="observed (logic)", color="green")
    
    # Plot the observed pattern's weighted K function (orange line)
    ax[1].plot(distances, K_values, label="observed (weighted)", color="orange")
    
    # Clean up labels and axes
    ax[1].set_xlabel("Distance")
    ax[1].set_ylabel("K(d) value")
    ax[1].legend()
    ax[1].set_xlim(0, max_dist)
    
    # Select which correction is applied
    if method == 0:
        ax[1].set_title(r"Weighted Ripley's $K(d)$ function")
    elif method == 1:
        ax[1].set_title(r"Weighted Besag's $K(d)$ function")
    elif method == 2:
        ax[1].set_title(r"Weighted WM's $K(d)$ function")
    else:
        ax[1].set_title(r"Unweighted $K(d)$ function")
    
    # Gather all the weight value for each event
    weight_list = np.zeros(len(points))
    for i in range(len(points)):
        weight_list[i] = np.sum(weights[i,:])
    
    # Plot the pattern itself on the right frame
    scatter = ax[2].scatter(points[:, 0], points[:, 1], c=weight_list, cmap='viridis')
    ax[2].set_title("Pattern")
    plt.colorbar(scatter, ax=ax[2], label='Weights')

    # Clean up labels and axes there, too
    ax[2].set_xticks([])
    ax[2].set_yticks([])
    ax[2].set_xticklabels([])
    ax[2].set_yticklabels([])
    
    # Save the figure
    file_path = os.path.join(full_save_path, 'Weighted_K_function.png')
    plt.savefig(file_path)
    plt.show()
    
    # Calculate the L-function values from the K-function values
    L_values, L_norm = weighted_ripley_l(K_values, K_norm)
   
    # Calculation of the Ripley's L function
    l_test = pointpats.l_test(points, support=40, hull=hull_pattern, keep_simulations=True, n_simulations=1000)

    # Parameters for the display
    fig, ax = plt.subplots(1, 3, figsize=(14, 4), gridspec_kw=dict(width_ratios=(4 ,6, 4)))
    
    # Gather all the weight value for each event
    unweight_list = np.zeros(len(points))
    for i in range(len(points)):
        unweight_list[i] = np.sum(unweight[i,:])
    
    # Plot the pattern itself on the left frame
    scatter = ax[0].scatter(points[:, 0], points[:, 1], c=unweight_list, cmap='viridis')
    ax[0].set_title("Pattern")
    plt.colorbar(scatter, ax=ax[0], label='No Weights')

    # Clean up labels and axes there, too
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_xticklabels([])
    ax[0].set_yticklabels([])

    # Plot all the simulations with very fine lines (black lines)
    ax[1].plot(l_test.support, l_test.simulations.T, color="k", alpha=0.01)
    
    # And show the average of simulations (blue line)
    ax[1].plot(
        l_test.support,
        np.median(l_test.simulations, axis=0),
        color="cyan",
        label="median simulation",
    )
    
    # And the observed pattern's L function (red line)
    ax[1].plot(l_test.support, l_test.statistic, label="observed", color="red")
    
    # Comparison with the analytic value
    ax[1].plot(distances, L_norm, label="observed (logic)", color="green")

    # Plot the observed pattern's weighted L function (orange line)
    ax[1].plot(distances, L_values, label="observed (weighted)", color="orange")

    # Clean up labels and axes
    ax[1].set_xlabel("Distance")
    ax[1].set_ylabel("L(d) value")
    ax[1].legend()
    ax[1].set_xlim(0, max_dist)
    
    # Select which correction is applied
    if method == 0:
        ax[1].set_title(r"Weighted Ripley's $L(d)$ function")
    elif method == 1:
        ax[1].set_title(r"Weighted Besag's $L(d)$ function")
    elif method == 2:
        ax[1].set_title(r"Weighted WM's $L(d)$ function")
    else:
        ax[1].set_title(r"Unweighted $L(d)$ function")

    # Gather all the weight value for each event
    weight_list = np.zeros(len(points))
    for i in range(len(points)):
        weight_list[i] = np.sum(weights[i,:])

    # Plot the pattern itself on the right frame
    scatter = ax[2].scatter(points[:, 0], points[:, 1], c=weight_list, cmap='viridis')
    ax[2].set_title("Pattern")
    plt.colorbar(scatter, ax=ax[2], label='Weights')

    # Clean up labels and axes there, too
    ax[2].set_xticks([])
    ax[2].set_yticks([])
    ax[2].set_xticklabels([])
    ax[2].set_yticklabels([])
    
    fig.tight_layout()
    
    # Save the figure
    file_path = os.path.join(full_save_path, 'Weighted_L_function.png')
    plt.savefig(file_path)
    plt.show()
    
# Function to display the K Ripley's function as the function 'display_Ripley_K' by adding the plot of the observed K-values under RC
def display_Ripley_K_L_weight_scaled(points, weights, unweight, max_dist, region, method, img_array_area,data_info, save_path, hull_pattern=None):
    """
    Parameters:
        points : A 2D numpy array of generated points (from the model)
        weights : A 2D numpy array of weights with w[i,j] the weight of the point i related to the point j
        unweight : A 2D numpy array of weights (not applied) with w[i,j] = 1 value
        max_dist : maximum distance for Ripley's K function
        region : list of for value of each side of the area (square or rectangle)
        method : Index of correction method selected
        img_array_area : Binary matrix to represent the data area
        data_info : additional information to specify the data (e.g., data_number, 'data2', etc.)
        hull_pattern: bounding box, scipy.spatial.ConvexHull, shapely.geometry.Polygon the hull used to construct a random sample pattern 
        save_path : path where the figure should be saved
    """
    
    # Ensure the save directory exists
    full_save_path = os.path.join(save_path, data_info)
    os.makedirs(full_save_path, exist_ok=True)
    
    # Implementation of the library area implicitly selected
    library_region = [min(points[:,0]), max(points[:,0]), min(points[:,1]), max(points[:,1])]

    # Calculate the weighted and unweighted Ripley K function
    distances, K_values, K_norm = weighted_ripley_k(points, max_dist, region, img_array_area, method, hull_pattern)

    # Calculate the Ripley K function with the library function called 'k_test'
    K_test = pointpats.k_test(points, support=40, hull=hull_pattern, keep_simulations=True, n_simulations=1000)

    # Parameters for the display
    fig, ax = plt.subplots(1, 3, figsize=(14, 4), gridspec_kw=dict(width_ratios=(4 ,6, 4)))
    
    # Gather all the weight value for each event
    unweight_list = np.zeros(len(points))
    for i in range(len(points)):
        unweight_list[i] = np.sum(unweight[i,:])
    
    # Plot the pattern itself on the left frame
    scatter = ax[0].scatter(points[:, 0], points[:, 1], c=unweight_list, cmap='viridis')
    ax[0].set_title("Pattern")
    plt.colorbar(scatter, ax=ax[0], label='No Weights')

    # Clean up labels and axes there
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_xticklabels([])
    ax[0].set_yticklabels([])
    
    # Calculate the region implicitly selected by the library 'pointpats' for the area
    library_area = (library_region[1] - library_region[0]) * (library_region[3] - library_region[2])

    # Creation of a new variable to store the updated simulations (with the change of area)
    K_updated_simulations = K_test.simulations * (hull_pattern.area / library_area)

    # Work on the updated simulations for the plotting (black lines)
    ax[1].plot(K_test.support, K_updated_simulations.T, color="k", alpha=0.01)

    # and print the median of all simulation (blue line)
    ax[1].plot(
        K_test.support,
        np.median(K_updated_simulations, axis=0),
        color="cyan",
        label="median simulation",
    )

    # Calculate as done before with the simulations the K-values with our own area selection (by using the ratio)
    K_observed_statistic = K_test.statistic * (hull_pattern.area / library_area)
        # and the function of the library K-function observed with the given pattern (red line)
    ax[1].plot(K_test.support, K_observed_statistic, label="observed", color="red")
    
    # Comparison with analytic value (green line)
    ax[1].plot(distances, K_norm, label="observed (logic)", color="green")
    
    # Plotting the weighted K-function of the observed pattern (orange line)
    ax[1].plot(distances, K_values, label="observed (weighted)", color="orange")
    
    # Display the normality check and distribution for a specific distance
    IC_distance = check_normality_and_display_first(K_test.support, K_updated_simulations, K_observed_statistic, data_info, save_path)
    
    if IC_distance:
        ax[1].axvline(IC_distance, color='r', linestyle='dashed', linewidth=1, label='distance inflexion')
    
    # Clean up labels and axes there
    ax[1].set_xlabel("Distance")
    ax[1].set_ylabel("K(d) value")
    ax[1].legend()
    ax[1].set_xlim(0, max_dist)
    
    # Select which correction is applied
    if method == 0:
        ax[1].set_title(r"Weighted Ripley's $K(d)$ function")
    elif method == 1:
        ax[1].set_title(r"Weighted Besag's $K(d)$ function")
    elif method == 2:
        ax[1].set_title(r"Weighted WM's $K(d)$ function")
    else:
        ax[1].set_title(r"Unweighted $K(d)$ function")
    
    # Gather all the weight value for each event
    weight_list = np.zeros(len(points))
    for i in range(len(points)):
        weight_list[i] = np.sum(weights[i,:])
    
    # Plot the pattern itself on the right frame
    scatter = ax[2].scatter(points[:, 0], points[:, 1], c=weight_list, cmap='viridis')
    ax[2].set_title("Pattern")
    plt.colorbar(scatter, ax=ax[2], label='Weights')

    # Clean up labels and axes there
    ax[2].set_xticks([])
    ax[2].set_yticks([])
    ax[2].set_xticklabels([])
    ax[2].set_yticklabels([])
    
    fig.tight_layout()
    
    # Save the figure
    file_path = os.path.join(full_save_path, 'Weighted_scaled_K_function.png')
    plt.savefig(file_path)
    plt.show()
    
    check_normality_and_display_first(K_test.support, K_updated_simulations, K_observed_statistic, data_info, save_path, display='on')
    
    # Get significant distances where the observed K function is greater than all simulations
    significant_distances = get_significant_distances(K_test.support, K_observed_statistic, K_updated_simulations)
    
    # Print significant distances
    print("Significant distances where observed K function is greater than all simulations:")
    print(significant_distances)
    
    # Get significant corrected distances where the observed K function is greater than all simulations
    significant_corrected_distances = get_significant_distances(K_test.support, K_values, K_updated_simulations)
    
    # Print significant distances
    print("Significant distances where observed weighted K function is greater than all simulations:")
    print(significant_corrected_distances)

    # Calculate the L-function values with/without correction weight
    L_values, L_norm = weighted_ripley_l(K_values, K_norm)
   
    # Calculate the L-function Ripley
    l_test = pointpats.l_test(points, support=40, hull=hull_pattern, linearized=True, keep_simulations=True, n_simulations=1000)

    # Parameters of the diplay
    fig, ax = plt.subplots(1, 3, figsize=(14, 4), gridspec_kw=dict(width_ratios=(4 ,6, 4)))
    
    # Gather all the weight value for each event
    unweight_list = np.zeros(len(points))
    for i in range(len(points)):
        unweight_list[i] = np.sum(unweight[i,:])
    
    # Plot the pattern itself on the left frame
    scatter = ax[0].scatter(points[:, 0], points[:, 1], c=unweight_list, cmap='viridis')
    ax[0].set_title("Pattern")
    plt.colorbar(scatter, ax=ax[0], label='No Weights')

    # Clean up labels and axes there
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_xticklabels([])
    ax[0].set_yticklabels([])

    # Recover and store the L-simulated value to apply after the area correction with the ratio
    l_updated_simulation_tmp = (l_test.simulations + distances)**2 * np.pi * (hull_pattern.area / library_area)
    # Recalculate the real L-values for simulations after the selection of our area
    l_updated_simulations = np.sqrt(l_updated_simulation_tmp / np.pi) - distances
    # Plot all the simulations with very fine lines (black lines)
    ax[1].plot(l_test.support, l_updated_simulations.T, color="k", alpha=0.01)
    
    # and the median of the simulations (blue line)
    ax[1].plot(
        l_test.support,
        np.median(l_updated_simulations, axis=0),
        color="cyan",
        label="median simulation",
    )
    
    # Then do the same done for the L-simulations with the observed L-values
    l_observed_tmp = (l_test.statistic + distances)**2 * np.pi * (hull_pattern.area / library_area)
    l_observed_statistic = np.sqrt(l_observed_tmp / np.pi) - distances
    # Plot our observed L-function related to the pattern (red line)
    ax[1].plot(l_test.support, l_observed_statistic, label="observed", color="red")
    
    # Comparison with the analytic value (green line)
    ax[1].plot(distances, L_norm - distances, label="observed (logic)", color="green")

    # Plot the weighted L-function of the observed pattern (orange line)
    ax[1].plot(distances, L_values - distances, label="observed (weighted)", color="orange")

    # Clean up labels and axes there
    ax[1].set_xlabel("Distance")
    ax[1].set_ylabel("L(d) value")
    ax[1].legend()
    ax[1].set_xlim(0, max_dist)
    
    # Select which correction is applied
    if method == 0:
        ax[1].set_title(r"Weighted Ripley's $L(d)$ function")
    elif method == 1:
        ax[1].set_title(r"Weighted Besag's $L(d)$ function")
    elif method == 2:
        ax[1].set_title(r"Weighted WM's $L(d)$ function")
    else:
        ax[1].set_title(r"Unweighted $L(d)$ function")

    # Gather all the weight value for each event
    weight_list = np.zeros(len(points))
    for i in range(len(points)):
        weight_list[i] = np.sum(weights[i,:])

    # Plot the pattern itself on the right frame
    scatter = ax[2].scatter(points[:, 0], points[:, 1], c=weight_list, cmap='viridis')
    ax[2].set_title("Pattern")
    plt.colorbar(scatter, ax=ax[2], label='Weights')

    # Clean up labels and axes there
    ax[2].set_xticks([])
    ax[2].set_yticks([])
    ax[2].set_xticklabels([])
    ax[2].set_yticklabels([])
    
    fig.tight_layout()
    
    # Save the figure
    file_path = os.path.join(full_save_path, 'Weighted_scaled_L_function.png')
    plt.savefig(file_path)
    plt.show()
    
    # Return the list of the significant distances where we can have a clustered pattern
    return significant_distances, significant_corrected_distances