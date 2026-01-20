import numpy as np
import shapely
import cv2

def img_contour_to_polygon(img,kernel_size,iterations):
    _, img_binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    img_dilation = cv2.dilate(img_binary, kernel, iterations=iterations) 
    img_fill = img_dilation.copy()
    h, w = img_fill.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(img_fill, mask, (0,0), 255) 
    img_fill = cv2.bitwise_not(img_fill)
    img_fill_dilation = cv2.bitwise_or(img_fill, img_dilation)
    img_erosion = cv2.erode(img_fill_dilation, kernel, iterations=iterations+1)
    img_erosion = np.uint8(img_erosion)
    mean_contour, _ = cv2.findContours(img_erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    mean_contour_points = mean_contour[0].reshape(-1, 2)
    relative_contour_points = np.array([(x / img_erosion.shape[1], y / img_erosion.shape[0]) for x, y in mean_contour_points])
    mean_contour_polygon = shapely.Polygon(relative_contour_points)
    return mean_contour_polygon
    
def img_contour_to_polygon_2(image):
    """
    Extracts the largest contour of a binary image and returns it as `shapely.polygon` 
    with relative coordinates.
    
    :param image: Grayscale image as NumPy array
    :return: `shapely.polygon` with coordinates normalized between 0 and 1
    """
    height, width = image.shape
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    relative_coords = [(point[0][0] / width, point[0][1] / height) for point in largest_contour]
    relative_polygon = Polygon(relative_coords)
    return relative_polygon
    
def IoU_poly(ref_poly, other_poly):    
    iou_polys=[]
    
    if isinstance(ref_poly_1, list):
        ref_poly_1 = Polygon(ref_poly_1)
    if isinstance(other_poly_1, list):
        other_poly_1 = Polygon(other_poly_1)
        
    for i in range(0,len(ref_poly)):       
        ref_poly_1=ref_poly[i]
        other_poly_1=other_poly[i][0]
        # Calculate intersection and union areas
        intersection_area = ref_poly_1.intersection(other_poly_1).area
        union_area = ref_poly_1.union(other_poly_1).area
        # Compute IoU
        iou_poly = intersection_area / union_area if union_area > 0 else 0  # Avoid division by zero
        iou_poly = round(iou_poly, 2)        
        iou_polys.append(iou_poly)
    return iou_polys

