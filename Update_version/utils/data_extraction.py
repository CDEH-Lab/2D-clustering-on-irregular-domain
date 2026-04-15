import os
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
from skimage import measure
from scipy.spatial import ConvexHull
from shapely.geometry import Point, Polygon
from pointpats import PointPattern

# Function to extract til file and display of it, with complements analysis
def image_tif_extraction(path, data_info, save_path):
    """
    Parameters:
        path : path of the tiff file to extract/analyze
        data_info : additional information to specify the data (e.g., 'data1', 'data2', etc.)
        save_path : path where the figure should be saved

    Returns:
        img_array_path : binary matrix of the image
    """
    
    # Ensure the save directory exists
    full_save_path = os.path.join(save_path, data_info)
    os.makedirs(full_save_path, exist_ok=True)
    
    # Image read from the path 
    img_path = tiff.imread(path)
    
    # Display of the image
    plt.title("Image of the area/events to study extracted from tiff file")
    plt.imshow(img_path)
    plt.xlabel("X Coordinates")
    plt.ylabel("Y Coordinates")
    
    # Determine the file name based on the presence of 'area' or 'dot' in the path
    if 'area' in path:
        file_name = 'area_data_file.png'
    elif 'dot' in path:
        file_name = 'dots_data_file.png'
    else:
        file_name = 'default_data_file.png'
    
    # Save the figure
    file_path = os.path.join(full_save_path, file_name)
    plt.savefig(file_path)
    plt.show()
    
    # Conversion of the image into a binary matrix
    img_array_path = np.array(img_path)
    
    # Obtain the different values informations (what value and how many)
    unique_values, counts = np.unique(img_array_path, return_counts=True)
    print(f"Unique values in the array: {unique_values}")
    print(f"Counts of each unique value: {counts}")
    
    # Switching value to substitute 255 to 1 (for binary value)
    img_array_path[img_array_path >= 1] = 1
    
    # Complementary analysis information related to the image        
    unique_values, counts = np.unique(img_array_path, return_counts=True)
    print(f"Unique values in the array after binarization: {unique_values}")
    print(f"Counts of each unique value after binarization: {counts}")
    
    # Return the binary matrix of the image
    return img_array_path

# Function to extract the outline of the zone
def outline_identification(img_array_area):
    """
    Perimeters:
        img_array_area : binary matrix of the image area

    Returns:
        outlines_circumference : array of the binary matrix with the coordintes of the outlines
    """
    
    # Find outlines at a constant value of 0.5 to find the boundary between 0 and 1
    outlines_circumference = measure.find_contours(img_array_area, level=0.5)

    # Adjust outlines for correct orientation by flipping the y-axis for the plotting
    adjusted_outlines = [np.array([[y, x] for x, y in outline]) for outline in outlines_circumference]

    # Plot the original image with the correct origin
    plt.imshow(img_array_area, cmap='gray', origin='upper')

    # Plot the outlines of the area
    for outline in adjusted_outlines:
        plt.plot(outline[:, 0], outline[:, 1], linewidth=2, color='blue')

    plt.title("Contours of the Area")
    plt.xlabel("X Coordinates")
    plt.ylabel("Y Coordinates")
    plt.show()
    
    # Return the first outlines array that could be modified later
    return outlines_circumference

# Function to figure out the events coordinates of the pattern
def dots_identification(img_array_dots, binary_value):
    """
    Parameters:
        img_array_dots : binary matrix of the image dots
        binary_value : the binary value of the events coordinates events
    Returns:
        coordinates : list of the coordinates of the events of the image extracted"""
        
    # Obtain the coordinates where an event is present (value 1)
    coordinates = np.where(img_array_dots == binary_value)
    coordinates = np.array(list(zip(coordinates[0], coordinates[1])))
    
    # Adjust y-coordinates to match the typical orientation
    adjusted_coordinates = np.array([coordinates[:, 1], img_array_dots.shape[0] - coordinates[:, 0]]).T
    
    # Plotting the points pattern of the image
    plt.scatter(adjusted_coordinates[:,0],adjusted_coordinates[:,1],c='blue',s=10,label='Data Points')
    plt.title("Point Process Extracted location")
    plt.xlabel("X Coordinates")
    plt.ylabel("Y Coordinates")
    plt.legend()
    plt.show()
    
    # Return coordinates of all the events
    return coordinates

# Function to display the area and the events inside it
def display_phenomena(img_array_area, img_array_dots, outlines, dots_coordinates, data_info, save_path):
    """
    Parameters:
        img_array_area : binary matrix of the image area
        img_array_dots : binary matrix of the image dots
        outlines : array of the binary matrix with the coordintes of the outlines
        dots_coordinates : list of the coordinates of the events of the image extracted
        data_info : additional information to specify the data (e.g., 'data1', 'data2', etc.)
        save_path : path where the figure should be saved
    """
    
    # Ensure the save directory exists
    full_save_path = os.path.join(save_path, data_info)
    os.makedirs(full_save_path, exist_ok=True)
    
    # Plot the area with the correct origin
    plt.imshow(img_array_area, cmap='gray', origin='upper')

    # Adjust outlines for correct orientation by flipping the y-axis for the plotting
    adjusted_outlines = [np.array([[y, x] for x, y in outline]) for outline in outlines]

    # Plot the contours
    for outline in adjusted_outlines:
        plt.plot(outline[:, 0], outline[:, 1], linewidth=2, color='blue', label='Contours')
        
    # Adjust y-coordinates to match the typical orientation
    adjusted_coordinates = np.array([dots_coordinates[:, 1], dots_coordinates[:, 0]]).T

    # Plot the points/events
    plt.scatter(adjusted_coordinates[:, 0], adjusted_coordinates[:, 1], c='red', s=10, label='Data Points')

    # Add title and labels
    plt.title("Contours and Points/Events")
    plt.xlabel("X Coordinates")
    plt.ylabel("Y Coordinates")
    plt.legend()
    plt.show()
    
    # Create a new figure
    plt.figure()

    # Plot the contours
    # Adjust contours for correct orientation by flipping the y-axis
    adjust_outlines = [np.array([[y, img_array_area.shape[0] - x] for x, y in outline]) for outline in outlines]

    for outline in adjust_outlines:
        plt.plot(outline[:, 0], outline[:, 1], linewidth=2, color='blue', label='Contours')

    # Switch of the coordinates values to correspond to the modeling outlines
    dots_coordinates = np.array([dots_coordinates[:, 1],img_array_dots.shape[0] - dots_coordinates[:, 0]]).T
    # Plot the points/events
    plt.scatter(dots_coordinates[:, 0], dots_coordinates[:, 1], c='red', s=10, label='Data Points')

    # Add title and labels
    plt.title("Contours and Points/Events")
    plt.xlabel("X Coordinates")
    plt.ylabel("Y Coordinates")
    plt.legend()
    # Save the figure
    file_path = os.path.join(full_save_path, 'display_phenomena.png')
    plt.savefig(file_path)
    plt.show()

# Function to correct/rectify the zone of the model     
def region_phenomena_determination(outlines_circumference,dots_coordinates):
    """
    Parameters:
        outlines_circumference : Outlines of our area data
        dots_coordinates : Coordinates of our events

    Returns:
        adjusted_outline : information of the outline of the area corrected to correspond to region
        fixed_coordinates : coordinates modified to correspond to the region limits
        region : box region of the area modified
        adjusted_region : region unmodified
    """
    
    # Assuming there's only one contour, if there are multiple you might need to handle them separately
    outline = outlines_circumference[0]

    # Find the bounding box of the contour
    min_y, min_x = np.min(outline, axis=0).astype(int)
    max_y, max_x = np.max(outline, axis=0).astype(int)
    
    # Extract the region of interest from the original image
    region = [0,max_x - min_x,0,max_y - min_y]
    adjusted_region = [min_x, max_x, min_y, max_y]

    # Coordinates modified to be inside the region4
    fixed_coordinates = np.array([dots_coordinates[:, 1], dots_coordinates[:, 0]]).T
    
    # Adjust filtered coordinates to the region of interest
    fixed_coordinates[:, 0] -= min_x
    fixed_coordinates[:, 1] -= min_y

    # Change the contour to correspond to the region of interest
    adjusted_outline = np.array([[y - min_x, x - min_y] for x, y in outline])
    
    # Return the outline of the area, the events coordinates in the correct orientation of the x and y axis and the regular area dimension associated
    return adjusted_outline, fixed_coordinates, region, adjusted_region

# Function to recover the outline of our model point pattern used to calculate the K-function value
def Hull_computation(area_region, dots_coordinates, img_array_area=None):
    # Switch of coordinates for the irregular area
    fixed_coordinates = np.array([dots_coordinates[:, 1], dots_coordinates[:, 0]]).T
    
    if img_array_area is None:
        # Bondary box (regular or irregular) for the implementation of K-function
        hull = Polygon([(area_region[0],area_region[2]),(area_region[1],area_region[2]),(area_region[1],area_region[3]),(area_region[0],area_region[3])])
        
        # Information linked to the point pattern model
        pp_r = PointPattern(dots_coordinates,window=hull)
        print("\n")
        pp_r.summary()
        
        # Return the regular boundary
        return hull, dots_coordinates
    else:
        # Figure out the contours of our area data
        outline = measure.find_contours(img_array_area, level=0.5)
        # Rectify the contour within the region of interest
        adjusted_outline = [np.array([[y, x] for x, y in outlines]) for outlines in outline]

        # Convert the outline into a list of points representing it
        points_list = []
        for outline in adjusted_outline:
            points_list.extend(outline)

        # Convert the list into array type with numpy
        points_array = np.array(points_list)

        # Create the ConvexHull from our events to create our outline's polygon
        hull_pattern_c = ConvexHull(points_array)

        # Generate the polygon which represents the irregular area of our model 
        polygon = Polygon(hull_pattern_c.points)

        # Check if all the points pattern are within or not the area
        points_inside = [Point(p).within(polygon) for p in fixed_coordinates]
        
        # Print the information of the points location compare the outline (out or in)
        if not all(points_inside):
            print("\nSome points are out of the hull")
            # Optionnal: print or not the points outside the area
            outside_points = [p for p, inside in zip(fixed_coordinates, points_inside) if not inside]
            print("Events out of the hull:", outside_points)
        else:
            print("\nAll events are inside the hull")
            
        
        # Information linked to the point pattern model
        pp_irr = PointPattern(dots_coordinates,window=polygon)
        pp_irr.summary()
        
        # Return the irregular bondary
        return polygon, fixed_coordinates