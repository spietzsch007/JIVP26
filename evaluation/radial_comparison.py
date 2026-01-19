#radial comparison of two masks for a specific center point
#(center points could be calculated with functions in distribution_center.py)
import shapely
import numpy as np
from general_functions import distance 

def line_at_angle(start_point, angle):
    """
    Create a shapley LineString.
    The Line goes from start_point to an end point 
    outside of the relativ image boundary .
    The end point will be calculated based on the angle 
    and longest distance from start_point to the boundary.
    
    Parameters
    ----------
    start_point : tuple
        relativ coordinates of the start point (x, y)

    angle : float
        angle (radian)
    
    Returns
    -------
    shapely.geometry.linestring.LineString
        Line from 
    """
    #get a radius slightly greater that the longest distance to the boundary
    radius = max(distance(start_point, (0,0)), distance(start_point, (0,1)),
                 distance(start_point, (1,0)), distance(start_point, (1,1))) + 0.01
    #calculate the relativ coordinates of the end point
    x = radius*np.cos(angle) + start_point[0]
    y = radius*np.sin(angle) + start_point[1]
    return shapely.geometry.LineString([start_point, (x,y)])


def intersection_radii(polygons_intersections, center):
    """
    Returns radii of start and end points of the intersections.
    Every intersection in polygons_intersections is the intetrsection
    between a shapely LineString and a shapely Polygon.

    Parameters
    ----------
    polygons_intersections : list
        List with one entry for each polygon (tuple).
        Every entry contains the polygon and all intersections with it.
        format: [(polygon1, array([intersection1, intersection2, ...])), ...]
    
    center : tuple
        relativ coordinates of the center (x, y)

    Returns
    -------
    list
        List with all intersection (ints) radii (r).
        Each sublist is for one polygon and contains 
        lists for all intersections with that polygon.
        That lists contain lists with radii where one 
        list belongs to one segment of the intersection.
    """
    all_r = []
    for polygon_intersection in polygons_intersections:
        #a new list for each intersection
        r = []
        #one polygon_intersection contains the polygon (polygon_intersection[0])
        #and the intersections (polygon_intersection[1])
        for intersection in polygon_intersection[1]:
            #calculate the radii
            distances = get_distances_to_center(intersection, center)
            #If the center is not inside of the polygon but there are intersections, insert [np.nan] to distances. 
            #Because of that, empty intersections, intersections for centers in the polygon 
            #and intersections for centers outside of the polygon can be distinguished.
            if not shapely.within(shapely.Point(center[0], center[1]), polygon_intersection[0]) and (len(distances[0]) != 0):
                distances.insert(0, [np.nan])
            r.append(distances)
        all_r.append(r)
    return all_r


def get_distances_to_center(geometries, center):
    """
    Calculates the distances to center for all
    points in all geometries in geometries.
    
    Parameters
    ----------
    geometries : shapely geometry
        could be a shapely Point, LineString,
        MultiLineString, MultiPoint or GeometryCollection.
        Geometries contains the geometries of one intersection 
        between a shapely Polygon and a LineString.
        
    center : tuple
        relativ coordinates of the center (x, y)

    Returns
    -------
    list
        List with all radii (r) for one intersection.
        That list contains lists with radii where one 
        list belongs to one segment of the intersection.
    """
    r = []
    #If geometries is a collection of geometries, loop througth them.
    if (type(geometries) != shapely.geometry.linestring.LineString) and (type(geometries) != shapely.geometry.point.Point):
        for geom in geometries.geoms:
            geom_r = []
            #append the distance for each point of one segment of an intersection to one list
            for point in shapely.get_coordinates(geom):
                #exclude the center point, but not single point intersections.
                if (point[0] != center[0] or point[1] != center[1]) or (type(geom) == shapely.geometry.point.Point):
                    geom_r.append(distance(center, point))
            r.append(geom_r)
    else:
        geom_r = []
        #append the distance for each point of the intersection to one list
        for point in shapely.get_coordinates(geometries):
            if (point[0] != center[0] or point[1] != center[1]) or (type(geometries) == shapely.geometry.point.Point):
                geom_r.append(distance(center, point))
        r.append(geom_r)
    return r


def intersection_deviation(radii_mask1, radii_mask2):
    """
    Radial deviation of the radii for two masks.
    
    Parameters
    ----------
    radii_mask1 : list
        Sublists containing start and end radius
        of one segment.
        No segments are represented as an empty
        sublist [].
        If the center was outside of the mask,
        the first sublist will contain np.nan.

    radii_mask2 : list
        Sublists containing start and end radius
        of one segment.
        No segments are represented as an empty
        sublist [].
        If the center was outside of the mask,
        the first sublist will contain np.nan.

    Returns
    -------
    list
        List with all radii (r) for one intersection.
        That list contains lists with radii where one 
        list belongs to one segment of the intersection.
    """
    #sort the lists of radii to make sure the closest segment to center is the first
    radii_mask1.sort(key=lambda r: np.mean(r))
    radii_mask2.sort(key=lambda r: np.mean(r))
    #The max_radius will be used als a normalization.
    if len(radii_mask1[-1]) == 0 and len(radii_mask2[-1]) == 0:
        max_radius = 0
    else:
        max_radius = np.nanmax(radii_mask1[-1] + radii_mask2[-1])
    #calculate the gaps between the segments of the mask
    gaps1 = radial_gaps(radii_mask1, max_radius)
    gaps2 = radial_gaps(radii_mask2, max_radius)
    #start to calculate the distance, were both mask are equal
    equal_distance = 0

    #Loop througth every segment of the second mask for every segment of 
    #the first mask and add the overlapping parts to equal_distance.
    for o1 in range(0, len(radii_mask1)):
        #taking care of segments that start at center
        if o1 == 0:
            min1 = 0
        else:
            min1 = segment_min(radii_mask1[o1])
        for o2 in range(0, len(radii_mask2)):
            #taking care of segments that start at center
            if o2 == 0:
                min2 = 0
            else:
                min2 = segment_min(radii_mask2[o2])
            #get the overlap as a intersection of to intervals
            overlap = intersection([min1, segment_max(radii_mask1[o1])], [min2, segment_max(radii_mask2[o2])])
            #add the overlap to equal_distances
            equal_distance = equal_distance + overlap
            
    #Do the same for the calculated gaps.
    #Masks are equal if they both exclude an interval.
    for o1 in range(0, len(gaps1)):
        for o2 in range(0, len(gaps2)):
            overlap = intersection(gaps1[o1], gaps2[o2])
            equal_distance = equal_distance + overlap

    if max_radius != 0:
        return equal_distance/max_radius
    else:
        return 1


def radial_gaps(radii, max_radius):
    """
    Radial gaps between the segments of radii.
    It will return gaps between zero and first 
    radius and between last radius and max_radius.
    
    Parameters
    ----------
    radii : list
        Sublists containing start and end radius
        of one segment.
        No segments are represented as an empty
        sublist [].
        If the center was outside of the mask,
        the first sublist will contain np.nan.

    max_radius : float
        Maximum of radii.
        If the last radius in radii is smaller
        than max_radius, a gap between them will be added.

    Returns
    -------
    list
        Sublists contain start and end radius
        of one gap.
        No gaps are represented as an empty
        sublist [].
    """
    gaps = []
    #if radii[0] is empty, there are no intersections
    if not radii[0] == []:
        #np.nan results in a radius of zero
        if np.nan in radii[0]:
            last = 0
        else:
            last = max(radii[0])
            
        for segment in radii[1:]:
            #If last is smaller than min(segment), 
            #there is a gap between them.
            if last < min(segment):
                gaps.append([last, min(segment)])
                last = max(segment)
        #Add a last gap if max_radius is bigger than the last radius.
        if max_radius > last:
            gaps.append([last, max_radius])
    return gaps


def intersection(interval1, interval2):
    """
    Intersection length of two intervals.
    
    Parameters
    ----------
    interval1 : list
        first interval
        format: [start_value, end_value]

    interval2 : list
        second interval
        format: [start_value, end_value]
        
    Returns
    -------
    float
        length of the intersection of the intervals.
    """
    intersection_min = max(min(interval1), min(interval2))
    intersection_max = min(max(interval1), max(interval2))
    return max(intersection_max - intersection_min, 0)


def segment_min(interval):
    """
    Minimum value of the interval.
    Takes care of radii interval specific
    cases. If the interval is empty or contains
    np.nan, zero is returned.
    
    Parameters
    ----------
    interval : list
        interval
        format: [start_value, end_value]
        
    Returns
    -------
    float
        Minimum radius of the interval
    """
    if len(interval) == 0:
        return 0
    elif np.nan in interval:
        return 0
    else:
        return min(interval)


def segment_max(interval):
    """
    Maximum value of the interval.
    Takes care of radii interval specific
    cases. If the interval is empty or contains
    np.nan, zero is returned.
    
    Parameters
    ----------
    interval : list
        interval
        format: [start_value, end_value]
        
    Returns
    -------
    float
        Maximum radius of the interval
    """
    if len(interval) == 0:
        return 0
    elif np.nan in interval:
        return 0
    else:
        return max(interval)


def plot_coordinates_of_nth_intersections(n, radii_sections, angles):
    """
    Function to get coordinates of one radial section to a pyplot compatible
    form.
    
    Parameters
    ----------
    n : int
        Intersection points of the nth intersection will be plottet.

    radii_sections : list
        Each sublist is for one section.
        The mask is partitioned into sections with
        the same amount of intersection points.
        The sections must be in the same order
        as angles.

    angles : list
        List of all angles.
        
        
    Returns
    -------
    list, list
        Angles and radii for all sections.
        Every section is in one sublist.
    """
    all_ang = []
    all_r_min = []
    all_r_max = []
    #number of angles already sorted to fitting radii
    last_a = 0
    for section in range(0, len(radii_sections)):
        ang = []
        order_r_min = []
        order_r_max = []
        for a in range(0, len(radii_sections[section])):
            #if the nth order excists, append coordinates
            if n < len(radii_sections[section][a]):
                ang.append(angles[last_a + a])
                order_r_min.append(radii_sections[section][a][n][0])
                order_r_max.append(radii_sections[section][a][n][-1])
        last_a = last_a + len(radii_sections[section])
        all_ang.append(ang)
        all_r_min.append(order_r_min)
        all_r_max.append(order_r_max)
    return all_ang, all_r_min, all_r_max


def mean_radii_at_angle(radii_at_angle):
    """
    Average radii of masks at one specific 
    angle.
    
    Parameters
    ----------
    radii : list
        Sublists containing start and end radius
        of one segment.
        No segments are represented as an empty
        sublist [].
        If the center was outside of the mask,
        the first sublist will contain np.nan.

    angle_index : int
        Index of the sublist for a sepecific angle
        in a list of radii for one mask.
        
    Returns
    -------
    list
        List with lists for each intersection segment.
        In every list are two sublists with the mean and 
        standard deviation values of the closer and the further 
        intersection point.
    """
    #find the highest order for all masks in radii for one angle
    max_order = get_max_order(radii_at_angle)
    #get the global mean for each order
    order_means = global_order_means(radii_at_angle, max_order)
    #Initilize the Array, that includes all points for each order.
    #There are 2 intersection points per order and len(radii) masks.
    ordered_points = np.full((max_order + 1, 2, len(radii_at_angle)), np.nan)
    #calculate and insert the intersection points for all masks at one angle
    for m in range(0, len(radii_at_angle)):
        #sort the intersection points -> first order is closest to mid point
        radii_at_angle[m].sort(key=lambda r: np.mean(r))
        #if there is no intersection: no orders can be inserted
        if (len(radii_at_angle[m]) == 1) & (len(radii_at_angle[m][0]) == 0):
            left_orders = 0
        #else the number of orders, that should be inserted is equal to the list length at angle_index position in m-th mask
        else:
            left_orders = len(radii_at_angle[m])
        #calculate all posible combinations to insert the left_orders
        combinations = positions(0, max_order, left_orders)
        #get the sum of distances to the order_means to find the best combination
        distances = [np.nansum([abs(order_means[combination[i]] - np.nanmean(radii_at_angle[m][i])) for i in range(0, len(combination))]) for combination in combinations]
        #len(distances) == 0 means there is nothing to insert
        if len(distances) != 0:
            #the best combination is the combination with lowest deviation to the order means
            best_combination = combinations[distances.index(min(distances))]
            start_order = 0
            for mask_order in range(0, len(best_combination)):
                #If best_combination only includes 0: the end_order must be 1, otherwise the 0th order will be lost because of range(0, 0) in the next for-loop.
                #This is not a problem if best_combinations includes a second element, because 0 will be included as start_order.
                if len(best_combination) == 1 and best_combination[0] == 0:
                    end_order = 1
                else:
                    end_order = best_combination[mask_order]
                
                for order in range(start_order, end_order):
                    #If order is in best_combination, the mask order for that combination should be inserted at that position.
                    if order in best_combination:
                        ordered_points[order][0][m] = min(radii_at_angle[m][best_combination.index(order)])
                        ordered_points[order][1][m] = max(radii_at_angle[m][best_combination.index(order)])
                    #If it is not on best_combinations, fill up with the better fitting mask order.
                    #If the last mask order is closer to the mean of order, insert it.
                    elif abs(np.mean(radii_at_angle[m][max(0, mask_order - 1)]) - order_means[order]) < abs(np.mean(radii_at_angle[m][mask_order]) - order_means[order]):
                        ordered_points[order][0][m] = min(radii_at_angle[m][max(0, mask_order - 1)])
                        ordered_points[order][1][m] = max(radii_at_angle[m][max(0, mask_order - 1)])
                    #else insert the radii of mask_order
                    else:
                        ordered_points[order][0][m] = min(radii_at_angle[m][mask_order])
                        ordered_points[order][1][m] = max(radii_at_angle[m][mask_order])
                #set start_order to end_order for the next mask_order
                start_order = end_order

            #If start_order is <= max_order after inserting the last mask order,
            #there are still unfilled spaces.
            if start_order <= max_order:
                for order in range(start_order, max_order + 1):
                    #If order is in best_combination, the mask order for that combination should be inserted at that position.
                    #That could happen here, because the end_order is excluded in the loop above.
                    if order in best_combination:
                        ordered_points[order][0][m] = min(radii_at_angle[m][best_combination.index(order)])
                        ordered_points[order][1][m] = max(radii_at_angle[m][best_combination.index(order)])
                    else:
                        ordered_points[order][0][m] = min(radii_at_angle[m][mask_order])
                        ordered_points[order][1][m] = max(radii_at_angle[m][mask_order])

    #start the list of mean and std values with the first order values.
    order_mean = (np.nanmean(ordered_points[0][0]), np.nanmean(ordered_points[0][1]))
    order_std = (np.nanstd(ordered_points[0][0]), np.nanstd(ordered_points[0][1]))
    all_order_means = [[[order_mean[0], order_std[0]], [order_mean[1], order_std[1]]]]
    #append the other orders
    for order in range(1, len(ordered_points)):
        order_mean = (np.nanmean(ordered_points[order][0]), np.nanmean(ordered_points[order][1]))
        order_std = (np.nanstd(ordered_points[order][0]), np.nanstd(ordered_points[order][1]))
        #if the last order and the current order overlap within the standard deviation
        if round((all_order_means[-1][1][0] + all_order_means[-1][1][1]), 15) >= round((order_mean[0] - order_std[0]), 15):
            #If the mean and the standard deviation of the last order are equal 
            #and there is only one order in all_order_means till now,
            #than the last order was order 0. 
            #In that case the lower intersection point must be changed to the new orders upper point.
            if (all_order_means[-1][1][0] == all_order_means[-1][0][0] and all_order_means[-1][0][1] == all_order_means[-1][1][1] and (len(all_order_means) == 1)):
                all_order_means[-1][0][0] = order_mean[1]
                all_order_means[-1][0][1] = order_std[1]
            #change the upper point to the new upper point to fuse the orders.
            all_order_means[-1][1][0] = order_mean[1]
            all_order_means[-1][1][1] = order_std[1]
        else:
            all_order_means.append([[order_mean[0], order_std[0]], [order_mean[1], order_std[1]]])
                
    return all_order_means


def get_max_order(radii_at_angle):
    """
    Get the highest order of intersections
    in radii at one angle.
    
    Parameters
    ----------
    radii : list
        List with one list per mask.
        Sublists are containing list for 
        each angle and in that list there
        are lists with radii for every intersection
        segment.

    angle_index : int
        Index of the sublist for a sepecific angle
        in a list of radii for one mask.
        
    Returns
    -------
    int
        highest order of intersections
        
    """
    max_order = 0
    for mask in radii_at_angle:
        if len(mask) > max_order:
            max_order = len(mask)
    return max_order - 1


def global_order_means(radii_at_angle, max_order):
    """
    Mean radii of all orders.
    The points are averaged from all masks.
    
    Parameters
    ----------
    radii : list
        List with one list per mask.
        Sublists are containing list for 
        each angle and in that list there
        are lists with radii for every intersection
        segment.

    angle_index : int
        Index of the sublist for a sepecific angle
        in a list of radii for one mask.

    max_order : int
        highest order of intersections
        
    Returns
    -------
    list
        list with the mean radii
        
    """
    order_means = []
    for order in range(0, max_order + 1):
        order_points = []
        for mask in radii_at_angle:
            if order < len(mask):
                order_points.extend(mask[order])
        order_means.append(np.mean(order_points))
    return order_means


def positions(start_position, max_order, left_orders):
    """
    All possible position combinations.
    The left_orders are must keep their order.
    Only positions between start_position and max_order are allowed.
    
    Parameters
    ----------
    start_position : int
        first possible position

    max_order : int
        highest order of intersections
        and last possible position

    left_orders : int
        number of orders that should be positioned
        
    Returns
    -------
    list
        list with all possible combinations
        
    """
    combinations = []
    #if left_orders > 1, use the function recursiv till only one is left
    if left_orders > 1:
        for position in range(start_position, (max_order - left_orders) + 2):
            next_order_combinations = positions(position + 1, max_order, left_orders - 1)
            new_combinations = [[position] + next_order_position for next_order_position in next_order_combinations]
            combinations.extend(new_combinations)
    #if only one order is left stop the recursion
    elif left_orders == 1:
        #If there is more than one position for the last order, extend a list with all left positions.
        if start_position != (max_order - left_orders + 1):
            combinations.extend([[position] for position in range(start_position, (max_order - left_orders) + 2)])
        else:
            combinations.append([max_order - left_orders + 1])
    return combinations


def mean_mask_uncertainty(mean_mask_points):
    """
    Get the uncertainty bands and the mean mask from
    mean_radii_at_angle outputs.
    
    Parameters
    ----------
    mean_mask_points : list
        list of mean_radii_at_angle outputs.
        Each sublist 
        
    Returns
    -------
    list, list, list
        First list contains the radii for the lower sigma uncertainty band.
        Second list contains the radii for the mean mask.
        Third list contains the radii for the upper sigma uncertainty band.
        Each list contains lists for segments of different order.
        In that lists the radii are saved.
    """
    r_min = []
    r_mean = []
    r_max = []
    last = np.nan
    for angle in mean_mask_points:
        flat_r_min = []
        flat_r_mean = []
        flat_r_max = []
        for order in angle:
            if (order[0][0] + order[0][1]) >= (order[1][0] - order[1][1]):
                flat_r_mean.append([order[0][0], order[1][0]])
                flat_r_min.append([max(order[0][0] - order[0][1], 0), np.nanmean([order[0][0], order[1][0]])])
                flat_r_max.append([np.nanmean([order[0][0], order[1][0]]) ,order[1][0] + order[1][1]])
            else:
                flat_r_mean.append([order[0][0], order[1][0]])
                flat_r_min.append([max(order[0][0] - order[0][1], 0), max(order[1][0] - order[1][1], 0)])
                flat_r_max.append([order[0][0] + order[0][1], order[1][0] + order[1][1]])
        number_of_radii = sum([len(order) for order in flat_r_mean])
        if last == number_of_radii:
            r_min[-1].append(flat_r_min)
            r_mean[-1].append(flat_r_mean)
            r_max[-1].append(flat_r_max)
        else:
            r_min.append([flat_r_min])
            r_mean.append([flat_r_mean])
            r_max.append([flat_r_max])
        last = number_of_radii
    return r_min, r_mean, r_max


def old_mean_mask_uncertainty(mean_mask_points):
    """
    Get the uncertainty bands and the mean mask from
    mean_radii_at_angle outputs.
    
    Parameters
    ----------
    mean_mask_points : list
        list of mean_radii_at_angle outputs.
        Each sublist 
        
    Returns
    -------
    list, list, list
        First list contains the radii for the lower sigma uncertainty band.
        Second list contains the radii for the mean mask.
        Third list contains the radii for the upper sigma uncertainty band.
        Each list contains lists for segments of different order.
        In that lists the radii are saved.
    """
    r_min = []
    r_mean = []
    r_max = []
    last = np.nan
    for angle in mean_mask_points:
        flat_r_min = []
        flat_r_mean = []
        flat_r_max = []
        for order in angle:
            if order[0][0] + order[0][1] >= order[1][0] - order[1][1]:
                flat_r_mean.extend([np.nanmean([order[0][0], order[1][0]])])
                flat_r_min.extend([max(order[0][0] - order[0][1], 0)])
                flat_r_max.extend([order[1][0] + order[1][1]])
            else:
                flat_r_mean.extend([order[0][0], order[1][0]])
                flat_r_min.extend([max(order[0][0] - order[0][1], 0), max(order[1][0] - order[1][1], 0)])
                flat_r_max.extend([order[0][0] + order[0][1], order[1][0] + order[1][1]])
        if last == len(flat_r_mean):
            r_min[-1].append(flat_r_min)
            r_mean[-1].append(flat_r_mean)
            r_max[-1].append(flat_r_max)
        else:
            r_min.append([flat_r_min])
            r_mean.append([flat_r_mean])
            r_max.append([flat_r_max])
        last = len(flat_r_mean)
    return r_min, r_mean, r_max

