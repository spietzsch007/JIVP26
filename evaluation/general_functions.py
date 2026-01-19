#some general functions including reading of YOLO txt files,
import numpy as np
import shapely
import os


def cartesian_coordiantes(polar_coordinates, center, size = (1,1)):
    """
    read mask from YOLO txt files
    
    Parameters
    ----------
    polar_coordinates : list
        Polar coordinates of the mask.
        index 0: list of angles
        index 1: list of radii

    center : tuple
        Coordinate origin

    size : tuple
        Size of the cartesian image.
        Could be used to directly scale the coordinates.
    
    Returns
    -------
    list
        Cartesian coordinates of the mask.
        index 0: x coordinates
        index 1: y coordinates
    """
    x = []
    y = []
    for i in range(0, len(polar_coordinates[0])):
        x.append((polar_coordinates[1][i]*np.cos(polar_coordinates[0][i]) + center[0])*size[0])
        y.append((polar_coordinates[1][i]*np.sin(polar_coordinates[0][i]) + center[1])*size[1])        
    return (x,y)


def read_masks_from_txt(masks_paths):
    """
    read mask from YOLO txt files
    
    Parameters
    ----------
    masks_paths : list
        List of paths (str) to the files. 
    
    Returns
    -------
    list, list
        Classes and points of the masks.
        Both lists contain one list per image and in that list one element per mask.
        For classes that elements are numbers that define the class.
        For points that elements are numpy arrays (numpy.ndarray) with [x, y] points.
    """
    all_masks = []
    all_classes = []
    for mask_path in masks_paths:
        with open(mask_path, 'r') as file:
            lines = file.readlines()
        file.close()
        classes = []
        masks = []
        for line in lines:
            str_data = line.replace('\n', '').split(' ')
            classes.append(int(str_data[0]))
            masks.append(np.array(str_data[1:], dtype = np.float64).reshape(int((len(str_data) - 1)/2), 2))
        all_masks.append(masks)
        all_classes.append(classes)
    return all_classes, all_masks


def filter_by_classes(masks, classes, filter_classes):
    """
    filter masks by their class
    
    Parameters
    ----------
    masks : list
        List of lists.
        Every sublist is for one image and contains
        numpy arrays (numpy.ndarray) with [x, y] points.
        Each array is for one mask.

    classes : list
        List of lists.
        Every sublist is for one image and contains
        the classes of all masks.

    filter_classes : list
        list of classes that should be returned
    
    Returns
    -------
    list
        Filtered masks.
        List of lists.
        Every sublist is for one image and contains
        numpy arrays (numpy.ndarray) with [x, y] points.
        Each array is for one mask.
    """
    filtered_masks = []
    for img in range(0, len(masks)):
        for mask in range(0, len(masks[img])):
            #If the class of the mask of img is in filter_classes, 
            #it should be returned.
            if classes[img][mask] in filter_classes:
                filtered_masks.append(masks[img][mask])
    return filtered_masks


def polygons_from_coordinates(masks):
    """
    Create shapely polygons from masks.
    The rectangular box around the polygon
    will be returned too.
    
    Parameters
    ----------
    masks : list
        List of numpy.ndarray.
        Each array is for one mask 
        and contains [x, y] points.
    
    Returns
    -------
    list, list
        index 0: list of created shapely polygons
        index 1: list of box areas
    """
    polygons = []
    areas = []
    for mask in masks:
        if len(mask) == 0:
            polygons.append(shapely.Polygon(mask))
            areas.append(None)

        else:
            #coordinates of corner points of the box
            min_coords = np.min(mask, axis = 0)
            max_coords = np.max(mask, axis = 0)
            areas.append((min_coords[0], max_coords[0], min_coords[1], max_coords[1]))
            #create a polygon if it is possible
            try:
                #polygon with less than 4 points will fail
                polygons.append(shapely.Polygon(mask))
            except:
                areas.append(None)
            
    return polygons, areas


def convert_YOLO_mask_to_xy(mask):
    """
    converts a YOLO mask (format: xy)
    to format x_list, y_list
    
    Parameters
    ----------
    mask : numpy.ndarray
        YOLO mask in xy format
    
    Returns
    -------
    list, list
        converted mask coordinates
        index 0: x coordinates (list of floats)
        index 1: y coordinates (list of floats)
    """
    #convert the coodinates to x and y lists.
    x = []
    y = []
    for xy in mask:
        x.append(xy[0])
        y.append(xy[1])
    return x, y


def distance(point1, point2):
    """
    2D Euclidean distance between two points
    
    Parameters
    ----------
    point1 : tuple
        first point
        format: (x, y) 
        x,y coordinates (float)

    point2 : tuple
        second point
        format: (x, y) 
        x,y coordinates (float)
    
    Returns
    -------
    float
        Euclidean distance between the points
    """
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def find_central_mask_save_as_yolo_txt(predictions, save_path):
    """
    Finds the most central mask from a list of predictions.
    If multiple masks exist, the one closest to the center of the image is selected.
    """

    best_segmented_masks = []

    for i in range(len(predictions)):
        segmented_masks = predictions[i].masks
        img_size = predictions[i].orig_shape
        img_path = predictions[i].path
        img_name,_ = os.path.splitext(os.path.basename(img_path))
        save_file = save_path + img_name + "_YOLO.txt"
                
        # Ensure masks are present
        if segmented_masks is not None and len(segmented_masks.xy) > 0:
            # Handle the case when there is only one mask
            if len(segmented_masks.xy) == 1:
                best_mask = segmented_masks.xy[0]
                best_segmented_masks.append(best_mask)
            # Handle the case when there are multiple masks
            elif len(segmented_masks.xy) >= 2:
                image_center = np.array([img_size[1] / 2, img_size[0] / 2])  # [width, height]
                min_distance = float('inf')
                best_mask = None
                # Iterate over all masks and find the one closest to the center
                for mask in segmented_masks.xy:
                    if len(mask) > 0:
                        mask_center = np.mean(mask, axis=0)
                        distance = np.linalg.norm(mask_center - image_center)
                        if distance < min_distance:
                            min_distance = distance
                            best_mask = mask
                if best_mask is not None:
                    best_segmented_masks.append(best_mask)
                    
        with open(save_file, 'w') as f:
            f.write("0 ")  # YOLO-Klassenindex
            valid_points = [
                f"{point[0] / img_size[1]} {point[1] / img_size[0]}"
                for point in best_mask if isinstance(point, (list, tuple, np.ndarray)) and len(point) == 2
            ]
            f.write(" ".join(valid_points))  # Nur gültige Punkte schreiben
            f.write("\n")
                  
    return best_segmented_masks
