#Functions to calculate the center of a distribution in an image.
import numpy as np
import scipy.optimize as opt

def get_centers(images, center_rules):
    """
    Calculate centers for all images in images.
    The calculation depends on the defined center_rules.
    There are 3 valid options:
    gauss: get the center from a 2D gaussian
    geometric: get the geometric center of a polygon
    img: get the mid of the image
    
    Parameters
    ----------
    images : list
        List of images.
        Each image is a numpy.ndarray with y-rows
        and x values in each row.
        It is posible to use multichannel images.
        Each x value would contain n-channels than.
        Those n-channels will be summed for the gaussian fit.

    center_rules : list
        List of rules for the center calculation.
        Each element is a list with the case (str) at the first index.
        Posible cases are: 'gauss', 'geometric', 'img'.
        All other cases will return (np.nan, np.nan) as center.
        The second index is a list of arguments for the specified case.
        If there is only one center_rule, all images will use that rule.
        If there is one rule per image, every image will use the rule at 
        the fitting index.
        Examples: center_rules = [['gauss', [0.5, 0.02, False, False]]] one rule for all
        center_rules = [['gauss', [1, False, False]], ['img'], ['geometric', [ex_poly]]
        rules for 3 images each with a different center calculation.
    
    Returns
    -------
    tuple
        Coordiantes of the center (floats)
    """
    #calculate centers for all images and save them in centers
    centers = []
    for i in range(0, len(images)):
        #if there is only one center rule, apply them to all images
        if len(center_rules) == 1:
            if center_rules[0][0] == 'gauss':
                try:
                    #get_gauss_in_image is used to get the center by an 2D gauss fit
                    popt = get_gauss_in_image(images[0], *center_rules[0][1])[0]
                    centers.append((popt[0], popt[1]))
                except:
                    centers.append((images[i].shape[0]/2, images[i].shape[1]/2))
            elif center_rules[0][0] == 'geometric':
                #the geometric center will be returned from a shapely polygon
                centers.append((center_rules[0][1][0].centroid.xy[0][0],
                                center_rules[0][1][0].centroid.xy[1][0]))
            elif center_rules[0][0] == 'img':
                #center of the image is at half shape for x and y direction
                centers.append((images[i].shape[0]/2, images[i].shape[1]/2))
            else:
                #if the case is not known, return nan center point
                centers.append((np.nan, np.nan))
        #else apply the ith rule to the ith image
        else:
            if center_rules[i][0] == 'gauss':
                try:
                    #get_gauss_in_image is used to get the center by an 2D gauss fit
                    popt = get_gauss_in_image(images[i], *center_rules[i][1])[0]
                    centers.append((popt[0], popt[1]))
                except:
                    centers.append((images[i].shape[0]/2, images[i].shape[1]/2))
            elif center_rules[i][0] == 'geometric':
                #the geometric center will be returned from a shapely polygon
                centers.append((center_rules[i][1][0].centroid.xy[0][0],
                                center_rules[i][1][0].centroid.xy[1][0]))
            elif center_rules[i][0] == 'img':
                #center of the image is at half shape for x and y direction
                centers.append((images[i].shape[0]/2, images[i].shape[1]/2))
            else:
                #if the case is not known, return nan center point
                centers.append((np.nan, np.nan))

    return centers


def get_gauss_in_image(img, ignore_fraction = 0.5,  averaged_fraction = 0.02, nested_channels = False, reverse = False):
    """
    2D gauss fit with scipy.optimize curve_fit.
    To fit the highest valued peak in the image, a fraction of values
    between the Minimum and Maximum could be ignored.
    The Minimum and Maximum is calculated as the average of 
    the averaged_fraction*(total number of points) smallest/biggest values.
    
    Parameters
    ----------
    img : numpy.ndarray
        Image with y rows with x values in each row.
        It is posible to use multichannel images.
        If the image contains lists of channel values instead
        of values, all channels are summed.
        (Set nested_channels to True in that case)

    ignore_fraction : float
        Fraction of values that will be ignored.
        ignore_fraction is a realtiv value (0 to 1), that represents 
        the value between background and peak.
        Values that are closer than ignore_fraction to the background
        are set to the background value.
        (optional, default = 0.5)

    averaged_fraction : float or int
        Fraction of the total number of points in the image,
        that is used to calculate the background and peak value.
        (optional, default = 0.02)

    nested_channels : bool
        Is true if elements in rows are not a number 
        and must be summed to one value.
        (optional, default = False)

    reverse : bool
        Is True if the peak to be fitted is neagtiv
        and False if it is positiv.
        (optional, default = False)
    
    Returns
    -------
    scipy.optimize curve_fit output, list
        Output of the fitting with scipy and a list of the modified values
    """
    #setup a grid with image shape
    x = np.linspace(0, img.shape[1], img.shape[1], endpoint = False)
    y = np.linspace(0, img.shape[0], img.shape[0], endpoint = False)
    x, y = np.meshgrid(x, y)
    
    #sum values of all channels if a value in the image is list and not a number.
    if nested_channels:
        added_values = np.array([sum(point) for row in img for point in row])
    else:
        added_values = np.array([point for row in img for point in row])

    #Find indices of n min and n max values.
    sorted_indices = np.argsort(added_values)
    n = round(img.shape[1]*img.shape[0]*averaged_fraction)
    min_indices = sorted_indices[:n]
    max_indices = sorted_indices[-n:]
    #Get the mean min and mean max values.
    added_max_values = 0
    added_min_values = 0
    for index in max_indices:
        added_max_values = added_max_values + added_values[index]
    for index in min_indices:
        added_min_values = added_min_values + added_values[index]
    mean_max_value = added_max_values/len(max_indices)
    mean_min_value = added_min_values/len(min_indices)

    #fit the gaussian depending on its sign
    if reverse:
        #For a reversed (negativ) peak, all values bigger than:
        #mean_max_value - the ignored fraction between max and min,
        #are set to mean_max_value.
        #That is used to fit to the highest value peak.
        fit_values = [i if i < mean_max_value - (mean_max_value - mean_min_value)*ignore_fraction
                      else mean_max_value for i in added_values]
        popt, pcov = opt.curve_fit(gaussian_2D, (x,y), fit_values, (img.shape[1]/2, img.shape[0]/2,
                                                                    img.shape[1]/10, img.shape[0]/10,
                                                                    max(fit_values),
                                                                    min(fit_values) - max(fit_values),
                                                                    0))
    else:
        #For a positiv peak, all values smaller than:
        #mean_min_value + the ignored fraction between max and min,
        #are set to mean_min_value.
        #That is used to fit to the highest value peak.
        fit_values = [i if i > mean_min_value + (mean_max_value - mean_min_value)*ignore_fraction
                      else mean_min_value for i in added_values]
        popt, pcov = opt.curve_fit(gaussian_2D, (x,y), fit_values, (img.shape[1]/2, img.shape[0]/2,
                                                                    img.shape[1]/10, img.shape[0]/10,
                                                                    min(fit_values),
                                                                    max(fit_values) - min(fit_values),
                                                                    0))
    return popt, pcov, fit_values


def gaussian_2D(coord, xo, yo, sigma_x, sigma_y, offset, amplitude, rotation_angle):
    """
    2D gauss function.
    
    Parameters
    ----------
    coord : numpy.ndarray, numpy.ndarray
        Coordiante grid. 
        The function will return values for all points on that grid.
        
    x0 : float
        expected value in x-direction
        
    y0 : float
        expected value in y-direction
        
    sigma_x : float
        standard deviation in x-direction
        
    sigma_y : float
        standard deviation in y-direction
        
    offset : float
        offset (background) value

    amplitude : float
        amplitude of the 2D gaussian

    rotation_angel : float
        The gaussian could be rotated by rotation_angel.
        rotation_angel 
    
    Returns
    -------
    numpy.ndarray
        Values of the function for each point in coords
    """
    #setup parameters and x, y coordinates
    x, y = coord
    a = (np.cos(rotation_angle)**2)/(2*sigma_x**2) + (np.sin(rotation_angle)**2)/(2*sigma_y**2)
    b = -(np.sin(2*rotation_angle))/(4*sigma_x**2) + (np.sin(2*rotation_angle))/(4*sigma_y**2)
    c = (np.sin(rotation_angle)**2)/(2*sigma_x**2) + (np.cos(rotation_angle)**2)/(2*sigma_y**2)
    #calculate all values
    gaussian = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
    return gaussian.ravel()