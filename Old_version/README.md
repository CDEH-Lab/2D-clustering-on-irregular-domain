--------------------------------------------
--     Spatial Point Pattern Analysis     --
--------------------------------------------

The repository is focus on a Biostatistic
field, which links mathematics (with 
statistics), and some computer science
knowledge.

You'll find files which will present some 
topics around this field in Python.

Currently, you are in the repository
named "Old_version" with 
4 notebooks that are here to execute the analysis
on each wanted experiment.

The latest files stored in this github are
the one where all the previous point pattern
analyses are applied on real experimental 
data. We study the kind of model that 
reprensents the proliferation of new cells
to respond to the death of some of them.
It happens on fly wings and we want to 
figure out if the pattern of the 
proliferation is random or clustered.

You can find the code and some results
in the files named :

"Drosophila_16h29_5h18_expXXX_(1).ipynb",

"Drosophila_16h29_5h18_expXXX_(1).ipynb",

"Drosophila_16h29_5h18_expXXX_(2).ipynb",

"Drosophila_16h29_10h18_expXXX_(2).ipynb"

--

expXXX : number of the experiment

(1) : 'ctrl' - control

(2) : 'rpr' - repair

--------------------------------------------

Execution :

To execute these code files, you just need to 
press "Execute ALL".

For the folders to analyze, you'll have to change
the path in the file by yourself. Here are the variables 
to change for the selection of your folders :

"directory_path" : Put the path of your folder 
which contains  your files
"save_path" : Change the path to store the 
analyses in the associated folder. 

This variable is 
located in differents functions :
- "display_Ripley_G"
- "display_Ripley_F"
- "display_Ripley_K"
- "display_Ripley_L"
- "display_Ripley_J"
- "display_Ripley_K_L_weight"
- "display_Ripley_K_L_weight_scaled"
- "cluster_and_display_points"
- "image_tif_extraction"
- "display_phenomena"

If you want to erase what you've generated, 
the last cell allows you to do it, you have to change 
the name the corresponding path folder in this cell before
executing it alone (or after the "Execute ALL"). 

To let you control the erase phase you must enter 
the following "password" : "KILL FOLDERS"
Otherwise, it will be unsuccessful.

Be sure for the execution to only have '.tif' file in your 
selected folders because in the exp192_ctrl.
A problem occured do to the fem_19 because 2 files was named like that
for the dots (one was a '.tif' file, the other useless '.part' file).

--

All the files named Drosophila can be adapted easily
for every experiment where inside the associated folder
you have '.tif' files with area and dots with the index 
expXXX and fem_XX_ in their name file. You just have to change 
the "directory_path" and the "save_path" to note overwrite on 
previous executions or not executing the one you want.

--------------------------------------------
--         Gabriel SOBCZYK-MORAN          --
--------------------------------------------
