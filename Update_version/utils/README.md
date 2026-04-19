------------------------------------------------------------------------------------------------------
------------------------------- Spatial Point Processes (Utils Folder) -------------------------------
------------------------------------------------------------------------------------------------------

This README is a complement of the main one (which is speaking about the notebook of the analysis). It will focus on every Python programs that are stored in the "utils" folder. Every program contains multiple functions which are gathered regarding their role (and actions). Each name affixed on these programs represents the main topic/role by the functions written within these programs.

This file will introduce all these programs, their actions, functions' role and a description of the programs and functions.

-----------------------------------------------------------

### 1. Program "__init__.py"

This program has a unique but essential utility. It connects our main notebook "Spatial_Analysis_Point_Pattern.ipynb" with the modules/programs store in "utils" repository.

It works like a bridge between these two distinct elements of the whole code. It's an empty program.

-----------------------------------------------------------

### 2. Program "clustering.py"

This program has one function that has to identify the possible cluster from the pattern provided based on the nearest-neighbor distance (or the distance 'd' of the Ripley's functions method) with the use of DBSCAN. 

- Function "cluster_and_display_points" : It's composed with 5 parameters, the pattern with "points", a list of distances 'd' (with "significant_distances") to apply the DBSCAN algorithm, a "method" regarding the Ripley's function selected and "data_info" and "save_path" to save the output.

First, the function checks if a "significant_distances" is provided. If 'YES', the saving path (folder) is created/found. Regarding the selected "method", the distance 'd' to use for DBSCAN is selected. Then, the clustering DBSCAN algorithm is launched based on the selected 'd' distance and the 'sample value' to form a cluster. After that, each point that composes a cluster receives the same color. The last action done by the function is to generate the figure and to save it.

-----------------------------------------------------------

### 3. Program "correction.py"

This program concentrates 10 functions that are all gathered on one large topic which is the correction weight to apply during the analysis (Ripley (RC), Besag (BC), Wiegand & Moloney (WM)). Some of them will apply RC & BC, others will be on WM and the lasts ones are on the calculations of the Ripley's function with these methods of correction.

- Function "calculate_distances" : This function takes in parameter only a list of every coordinates of the events' model in "points". With that, we retrieve the length of the list to count the number of events and use it to generate a matrix (n x n) to stored the (Euclidean) distance in it for all the events between them.

- Function "calculate_e1_e2" : The function takes in parameter the coordinates of one of the events of the model. It also receives the border (outline) data of the area (regular form) of study. The function starts by extracting the coordinates of the provided "point" and do the same with the interval of "region". Then the function determines "e1" and "e2" which are respectively the 2 direct shortest distances between the point and the borders of the area.

- Function "calculate_weight_Ripley" : The function applies the calculation of the weight of Ripley (w_rc) by using the coordinates of a provided point, the borders area with "region" and the distance 'd' selected for the Ripley's function K calculation (with the parameter "h"). The function calculates the "e1" and "e2" value with "calculate_e1_e2", then regarding the conditions between these 2 new values and "h", the Ripley correction's weight is calculated.

- Function "calculate_weight_Besag" : The function applies the calculation of the weight of Besag (w_bc) by using the coordinates of a provided point, the borders area with "region" and the distance 'd' selected for the Ripley's function K calculation (with the parameter "h"). The function calculates the "e1" and "e2" value with "calculate_e1_e2", then regarding the conditions between these 2 new values and "h", the Besag correction's weight is calculated.

- Function "calculate_weight_Wiegand_Moloney" : With 3 parameters (coordinates of an event in "point", the matrix of the whole image area that stores the study case area (irregular form) and even more outside of it in "img_array_area" and "h" the selected distance for the Ripley's function K calculation), the function executes the Wiegand & Moloney correction. It starts by retrieving the matrix shape (with "n_rows" & "n_cols"). After it, the regular area of the circle of diameter "2h" is highlighted and with it, the disk of radius "h" is identified within a matrix. From the point (center of the disk), a sub-area of the image matrix is extracted (and then the code is making sure than whole the disk is taken into account). The value "h" is adapt in int to match with the pixels value and include the entire disk. In the end the function find out the pixels that are in the disk area and in the study area at the same time. The weight will be the ratio of the part of pixels of the disk in the study area on the pixels that compose the whole disk.

- Function "compute_weights" : The function is calculating every pair of points of the model in a matrix (n x n). We have the list of point coordinates with "points", the selected distance 'd' for Ripley's function K in "h", the study of area with "region", the whole image area translated in the matrix "img_array_area" and then the "method" of correction selected. First, the function initializes the matrix of weights, then calculates every weight of all possible pairs of events with the next function "compute_w_ij".

- Function "compute_w_ij" : The weight of a given pairs of points "s_i" & "s_j" is calculated in this function regarding the selected method of correction. The function retrieves the coordinates of the pair of points, the distance between them is calculated. If this distance is upper the "h" one, the weight is normal (not modified) with "1". Otherwise, if it's under, the correction method selected will calculate the weight of this pair.

- Function "Correction_K_function" : With the coordinates list of the model in "points", the "region", the value "h", the matrix of the whole image in "img_array_area", the selected correction "method" and the given "hull", the function will calculate the corrected Ripley's function K, the normal one, the matrix of "weights" for each point with the application of corretion (and without with "w"). The function starts by calculating the pair distance, their associated weight, initializes the indicator of the Ripley's function K then calculates K. The calculation of the 2 kind of K is completed with the density in "hull" and the number of point that compose the pattern model.

- Function "weighted_ripley_k" : This function uses the previous one by calculating the Ripley's function K value for a linespace value of 'd' and not a single selected 'd' value.

- Function "weighted_ripley_l" : Same as "weighted_ripley_k" but for Ripley's function L this time.

This program allows us to calculate the Ripley's function K and L with and without any provided correction method (RC, BC, WM).

-----------------------------------------------------------

### 4. Program "data_extraction.py"

This program is here to extract all the useful data that can be found in the given TIFF files of 'areapouch' & 'dots' files. The model pattern, the borders of the stydy area and the matrix of the study area with be extracted.

- Function "image_tif_extraction" : We have the "path" of the file, its "data_info" to recognize the experiment and the "save_path" to save the plotting of the extracted data. It starts to read the image TIFF file (and identify regarding the path name), then after the image type identification, the matrix of the image is deducted and returned.

- Function "outline_identification" : With the matrix "img_array_area" the function has to deduct the outline of the study area. From the area TIFF file image, the circumference of the study area is identified between the neighbor value "0" and "1". From it, the coordinates points of this circumference is determined to be displayed. The circumference data is returned.

- Function "dots_identification" : With the correct given "binary_valeu" and the dots that compose the model matrix given, the function will return the list of events' coordinates. If the values matrix match with the given "binary_value" is a event otherwise no. After it, the dots coordinates is returned.

- Function "display_phenomena" : The function is here to highlight all the extracted data and display it.

- Function "region_phenomena_determination" : The function normalized the area and dots to ensure that the origine is (0,0).

- Function "Hull_computation" : The function stores the minimum data information to have the irregular area and the "fixed_coordinates". It's important to calculate the Ripley's function K.

-----------------------------------------------------------

### 5. Program "main_analysis.py"

This program connects all the others present in "utils" folder with the main notebook. It contains the 2 main functions of the analysis (one doing the analysis and the other the extraction of data used to identify the experiment currently analyzed). In the library part the function needs the module of the other programs.

- Function "analysis_function" :  The function is doing the analysis of a provided experiment. With the "path_area" (path to reach the area image of the experiment), "path_events" (path to reach the model events image of the experiment), "data_number" which is the ID of the experiment, the selected "binary_value" to extract the area and the model and the path to save the output analysis, the function will first extract all the necessary data from the TIFF image file (area, dots, boundary...). It will also diplay them and generate new variables which are more adapt for the analysis. Then Ripley's function K is calculated, the nn-distance information also done, all the kind of displays are called (NND, kde, K, G, L, H...). Then regarding the results (based on the list of distance 'd' in which the point pattern can be considered as clustered is above 2 or not) a conclusion is taken with a display of the the cluster based on one the distance 'd' of the list.

- Function "extract_data_info" : From the image paths, this function must identify and therefore codify the name of the experiment sent to analyze.

-----------------------------------------------------------

### 6. Program "spatial_analysis.py"

This program retrieves all the statistical spatial analyses presented in the report excepted the Ripley's functions (G, K, L, H). We will have the nearest neighbor distances, Skellam, Clark-Evans, quadrats, p-value. The last function will be related Ripley's function (G normally or all).

- Function "unique_nearest_neighbor_distances" : With the provided "points" pattern, the function determines the unique NND of each point. It starts by calculating all pairwise distances then generates a square matrix from it (d_ij == d_ji). Then the minimum distance of a point 'i' is seeking within all the other points, 'j' is determined to be the NND with 'i' and all this stuff is returned.

- Function "Clark_Evans_Skellam_function" : The function will calculate the Skellam's statistic and will do the Clark-Evans test in the same function. We need in parameters, the "points" pattern of the model, the NND "nn_distances", the density model with "lambda" and a "method" value to determine if we want to do one test out of two (and which one) or both. 

For "method == 0 or method == 2", we have the Skellam's statistic done :

First, we learn the length of NND find out, with it we engage our loop. The Skellam's statistic formula is applied with first the sum of NND square product in the loop and them multiply by its constant. Then, this value and the critical one calculated with the degree of freedom is returned.

For "method == 1 or method == 2", we have the Clark-Evans test done :

The Clark-Evans test is effectuated based on the formula. Starts with the CLT parameters, and the mean CLT and then the normalized mean CLT.

(These function could be a bit rewritten to optimize the output of method 2).

- Function "Clark_Evans_simulation" : Due to the fact that the Clark-Evans test don't use all the points pattern of the model this function will select them randomly N times and returns the mean values of the previous function.

- Function "p_value_behaviour" : With the Clark-Evans output value and the "alpha" value this function is deployed to determined is the CSR Hypothesis is rejected or not for the model.

- Function "quadrats_method" : With the "points" pattern of the model provided, the quadrats method is applied and displayed. It starts by generating the quadrats and then calculates the chi**2, degree of freedom, threshold for model identification of p-value method. Regarding these values conclusion on the model is done (it's only focus on regular area).

- Function "get_significant_distances" : This last function of the program is trying to highlight the distances 'd' possible in which Ripley's functions (G, K, L, H) value at distance 'd' is above all simulations of the Ripley's function value at the same distance for any random pattern regarding our selected experiment pattern.

-----------------------------------------------------------

### 7. Program "statistical_analysis.py"

This program contains a single huge function that displays the Ripley's function selected.

-----------------------------------------------------------

### 8. Program "visualization.py"

This last Python program gathers all the function that prints elements of the analysis.

-Function "plot_kde_with_weights" : The function plots the kernel density of the pattern model regarding the selected correction method (RC, BC, WM or none).

- Function "display_kernel_weights" : The function display the weight repartition of the events pattern model (is useful to check the good application of the correction method in particular WM).

- Function "display_kernel_weights_bis" : Same function as the previous one but here the outline of the study area is displayed too.

- Function "display_spatial_points_analysis" : This function displays the NND of the pattern with blue dots for points pattern, red lines for NND (between 'i' and 'j') and green circles to confirm thant expect the NN point, no point is within the area of the circle.

- Function "display_Ripley_G" : This function executes entirely the Ripley's function G and displays it (independant of the area).

- Function "display_Ripley_F" : This function executes entirely the Ripley's function F and displays it (dependant of the area).

- Function "display_Ripley_K" : This function executes the Ripley's function K and displays it (independant of the area, here it's done on regular instead of the irregular/general case area).

- Function "display_Ripley_L" : This function executes the Ripley's function L and displays it (independant of the area, here it's done on regular instead of the irregular/general case area).

- Function "display_Ripley_J" : This function executes entirely the Ripley's function J and displays it (dependant of the area).

- Function "display_Ripley_K_L_weight" (customization of the previous Ripley's funcion K & L): This function displays the Ripley's function K & L with the application of the selected weight, also displays the model pattern with and without correction besides the Ripley's function (dependant of the area, here the simulations of points for a random pattern are done blindly on probably a regular area instead of a irregular/general case area). Here the value lambda for the simulation isn't controled.

- Function "display_Ripley_K_L_weight_scaled" (customization of the previous Ripley's funcion K & L): This function displays the Ripley's function K & L with the application of the selected weight, also displays the model pattern with and without correction besides the Ripley's function (dependant of the area, here the simulations of points for a random pattern are done in a more precise regular area than before, it's the closest version of an irregular/general case area). Here the value lambda for the simulation is well controled and adapt regarding our irregular region area & the precise regular region area.

-----------------------------------------------------------


------------------------------------------------------------------------------------------------
------------------------------------- Gabriel SOBCZYK-MORAN ------------------------------------
------------------------------------------------------------------------------------------------
