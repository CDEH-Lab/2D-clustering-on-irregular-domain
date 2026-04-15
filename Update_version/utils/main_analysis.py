import os
import numpy as np
import re
from .data_extraction import image_tif_extraction, outline_identification, dots_identification, display_phenomena, region_phenomena_determination, Hull_computation
from .spatial_analysis import unique_nearest_neighbor_distances, quadrat_method, p_value_behaviour, get_significant_distances
from .correction import Correction_K_function, weighted_ripley_k, weighted_ripley_l
from .visualization import plot_kde_with_weights, display_kernel_weights, display_kernel_weights_bis, display_spatial_points_analysis, display_Ripley_G, display_Ripley_K_L_weight_scaled, Ripley_dispaly_all
from .clustering import cluster_and_display_points

# Main function to analyze the data and display the results of the spatial analysis with/without correction method applied
def analysis_function(path_area, path_events, data_number, binary_value=0, method=2, save_path=None):
    # Extraction of the area information and plotting them
    img_array_area = image_tif_extraction(path_area,data_number,save_path)
    area_boundary = outline_identification(img_array_area)
    
    # Extraction of the dots model information and plotting them
    img_array_dots = image_tif_extraction(path_events,data_number,save_path)   
    dot_coordinates = dots_identification(img_array_dots,binary_value=binary_value)
    
    # Display the area and the events in the same graph
    display_phenomena(img_array_area,img_array_dots,area_boundary,dot_coordinates,data_number,save_path)
    
    # Extraction of the regular area of the outlines recover from the phenomena to study
    area_circumference, dots_coordinates, area_dim, adjusted_dim = region_phenomena_determination(area_boundary,dot_coordinates)
    
    # K-function radius h selected to check the well implementation of the corrected methods
    h = 0.2 * min(area_dim[1],area_dim[3])
    
    # Bondary box (regular or irregular) for the implementation of K-function
    hull, _ = Hull_computation(area_dim,dots_coordinates)
    hull_irr, fixed_coordinates = Hull_computation(area_dim,dot_coordinates,img_array_area)
    
    print("\nAnalysis of our data pattern\n")
    
    # Calculation of the analytic K-function value with associated weight related to the method selected
    K_corr_WM, K_norm_irr, w_wm, _ = Correction_K_function(fixed_coordinates, adjusted_dim, h, img_array_area, method=2, hull=hull_irr) 
    K_corr_BC, _, w_bc, _ = Correction_K_function(dots_coordinates, area_dim, h, img_array_area, method=1, hull=hull)
    K_corr_RC, K_norm, w_rc, w = Correction_K_function(dots_coordinates, area_dim, h, img_array_area, method=0, hull=hull)

    # Printing of the K-function values (corrected or not)
    print(f"K-function value with WM's correction : {K_corr_WM}")
    print(f"K-function value without correction method : {K_norm_irr}")
    print(f"K-function value with Besag's correction : {K_corr_BC}")
    print(f"K-function value with Ripley's correction : {K_corr_RC}")
    print(f"K-function value without correction method : {K_norm}")

    # Calculation of the Nearest Neighbor Distances (NN-Distances) and their corresponding points indices
    nn_distances, indices_nn_d, pairs_points = unique_nearest_neighbor_distances(dots_coordinates)
    
    # Number of distances stored by NN-Distances list ('nn_distances', 'indices_nn_d')
    # m = len(nn_distances)
    # print(f"\nNombre de NN-Distances : {m}")
    # print(f"Nombre d'indices de ces distances : {len(indices_nn_d)}")
    # print(f"Nombre de couples d'indices de ces distances : {len(pairs_points)}")
    # print(f"Les données NN-distances uniques : {nn_distances}")
    # print(f"Les indices des ces NN-distances uniques : {indices_nn_d}")
    # print(f"Couples d'indices des nn-d distances : {pairs_points}")
    
    # Display of the model with weight correction possibly applied
    # Display of the kNN information
    display_spatial_points_analysis(dots_coordinates,area_dim,len(dots_coordinates),indices_nn_d,nn_distances,pairs_points)
    
    # Dispaly of the weight repatition in the model
    display_kernel_weights_bis(fixed_coordinates,w_wm,area_boundary)
    # display_kernel_weights(dots_coordinates,w_rc,area_dim,h)
    # display_kernel_weights(dots_coordinates,w_bc,area_dim,h)
    display_kernel_weights(dots_coordinates,w,area_dim,h)
    
    # Display of the quadrat method
    quadrat_method(dots_coordinates)
    
    # Detemination of the max value of kNN used in the plotting of K-function
    max_nn_d = max(nn_distances)
    
    # Display of the K an L functions with/without weight impact
    # display_Ripley_K_L_weight(fixed_coordinates,w_wm,w,max_nn_d,adjusted_dim,2,img_array_area,data_number,hull_pattern=hull_irr)
    # display_Ripley_K_L_weight(dots_coordinates,w_rc,w,max_nn_d,area_dim,0,img_array_area,data_number,hull_pattern=hull)
    # display_Ripley_K_L_weight(dots_coordinates,w_bc,w,max_nn_d,area_dim,1,img_array_area,data_number,hull_pattern=hull)
    # Ripley_dispaly_all(dots_coordinates, max_nn_d,data_number,hull=hull)
    sup_dist = display_Ripley_G(dots_coordinates,max_nn_d,data_number,save_path,hull_pattern=hull)
    
    print("Scaled area")
    
    # Display of the K and L functions with/without weight impact with the good area selection imput
    # display_Ripley_K_L_weight_scaled(dots_coordinates,w_rc,w,max_nn_d,area_dim,0,img_array_area,data_number,hull_pattern=hull)
    # display_Ripley_K_L_weight_scaled(dots_coordinates,w_bc,w,max_nn_d,area_dim,1,img_array_area,data_number,hull_pattern=hull)
    # display_Ripley_K_L_weight_scaled(fixed_coordinates,w_wm,w,max_nn_d,adjusted_dim,2,img_array_area,data_number,hull_pattern=hull)
    check_dist, corrected_dist =  display_Ripley_K_L_weight_scaled(fixed_coordinates,w_wm,w,max_nn_d,adjusted_dim,2,img_array_area,data_number,save_path,hull_pattern=hull_irr)
    
    # Comparison of the Kernel Density with/without the application of a method of correction
    # plot_kde_with_weights(dots_coordinates, w_rc, area_dim,0)
    # plot_kde_with_weights(dots_coordinates, w_bc, area_dim,1)
    plot_kde_with_weights(fixed_coordinates,w_wm, adjusted_dim,2)
    plot_kde_with_weights(dots_coordinates, w, area_dim,0)
    
    if len(sup_dist) < 2 and len(check_dist) < 2 and len(corrected_dist) < 2:
        cluster_and_display_points(fixed_coordinates,check_dist,1,data_number)
        print("Case 5:")
        print("This point pattern isn't a clustered model")
        return 5
    elif len(check_dist) < 2 and len(corrected_dist) >= 2:
        cluster_and_display_points(fixed_coordinates,corrected_dist,2,data_number)
        print("Case 3:")
        print("This point pattern could be a clustered model related to the correction method in K function")
        return 3
    elif len(sup_dist) < 2:
        cluster_and_display_points(fixed_coordinates,check_dist,1,data_number)
        print("Case 2:")
        print("This point pattern might be a clustered model related to the K function analysis")
        return 2
    elif len(corrected_dist) < 2:
        cluster_and_display_points(fixed_coordinates,sup_dist,0,data_number)
        print("Case 4:")
        print("This point pattern could be a clustered model related to the G function analysis")
        return 4
    else:
        cluster_and_display_points(fixed_coordinates,sup_dist,0,data_number)
        print("Case 1:")
        print("This point pattern is a clustered model")
        return 1
        
 
# Function to extract the name/label of the data   
def extract_data_info(file_path):
    # Define the regex patterns to match 'expXXX' and 'fem_XX' separately
    exp_pattern = r'exp\d+'
    fem_pattern = r'fem_\d+'
    
    # Search for the first occurrence of each pattern in the file path
    exp_match = re.search(exp_pattern, file_path)
    fem_match = re.search(fem_pattern, file_path)
    
    if exp_match and fem_match:
        exp_part = exp_match.group(0)
        fem_part = fem_match.group(0)
        return f"{exp_part}_{fem_part}"
    else:
        return None