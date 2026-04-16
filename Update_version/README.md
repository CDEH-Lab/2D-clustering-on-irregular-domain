                    ----------------------------------------------------
                    --------- Spatial Point Processes (Update) ---------
                    ----------------------------------------------------

This README is focused on the Update of the Spatial Point Processes Analysis Python notebook programmes stored in "Old_version" folder.

This update is done to optimize the comprehension of the overall code and to concentrate all the modifiable parameters of the whole code in only one cell (in one notebook instead of one notebook per experiment).

This file will explain and describe what happen and how the analysis is done.

-----------------------------------------------------

First, before going straight to the description, we introduce the architecture of the analysis' programmes :

- One main notebook : "Spatial_Analysis_Point_Pattern.ipynb"

This main notebook gather all the principle function and cells to make the analysis work. 
The first cell calls all the needed librairies including the modules (Python programmes) that came from "utils" folder.
The second cell stores all the modifiable parameters of the analysis :

- directory_path : Path of the repository to analyze (i.e. "16h29_5h18_exp185_ctrl"). It contains the TIFF files to analyze with the duo of 'dots' and 'areapouch'.
- binary_value : Value that allows the analysis & extraction functions to find out in the data the surface, outlines and events of an experiment that will be studied by different functions.
- fem_start : Value to determine which experiment we have to start because some of them aren't correctly defined for the analysis done by this code (such as exp_191_fem_1).
- method : Value to determine which type of correction you want to apply for the analysis (0 : Ripley (RC), 1 : Besag (BC), 2 : Wiegand & Moloney (WM)).
- save_path : Path of the repository in which the displays of the code will be stored automatically (Sorting the output data).
- n_simulations : Selected number of simulation (of random pattern) for the Ripley's functions (more this number is high, more it will be easy to identify the CSR Hypothesis condition on the Ripley's function displays. But too high will exponentially extend the duration of the code execution).
- alpha : Percentage value selected regarding Skellam's statistics and Clark-Evans value (details in tha associated report).

The third cell is the core of the analysis ("main" function). It starts by detecting all the experiments stored in each folder (from directory_path). A counter for the clustered and random case is respectively generated. Then, the function generates a loop to analyze every exepirement detected. In this loop, for an experiment, the area TIFF file and dots TIFF file are identified and extracted. After the verification of their existance, the analysis is lanched with "analysis_function" method. The cell after it launches the previous "main" function.

An optional cell (the last one) is here to clean the save_path directory created by the previous cell.

-----------------------------------------------------

The functions called and used in this notebook are stored in "utils" folder and their content is explained in the associated "README.md"

                    -------------------------------------------
                    ---------- Gabriel SOBCZYK-MORAN ----------
                    -------------------------------------------
