import math
import cv2
import numpy as np

def generate_fiducial_marker(outer_color_hsv=120, size=200, inner_square_ratio=1/4, outer_square_ratio=10/12):
    '''
    Creates a fiducial marker which works with the detection class in simple_featture_marker_detection.py
    The default color scheme is blue and yellow
    Takes four inputs:
    outer_color_hsv: The hsv color of the out score between 0 and 180 deg
    size: The size of the marker in pixels. Tip: Create a very large marker and use a standard printing feature to scale it to any paper size
    inner_square_ratio: How much space the inner square is going to take of the total marker
    outer_square_ratio: How much space the outer square is going to take of the total marker
    '''
    def gen_clean_square(square_size, color):
        color_img = np.full((square_size, square_size, 3), color, dtype=np.uint8)
        return color_img

    # Determine wheiter to use floor or ceil
    floor_size = math.floor(size * ((1 - outer_square_ratio)/2 + outer_square_ratio)) - math.floor(size * (1 - outer_square_ratio)/2)
    if floor_size == math.floor(size * outer_square_ratio):
        round_funtion = math.floor
    else:
        round_funtion = math.ceil

    # Create a single pixel in HSV spectum
    outer_color_hsv = np.full((1, 1, 3), [outer_color_hsv, 255, 255], dtype=np.uint8)
    # Use cv2 to convert from HSV to BGR
    outer_color_bgr = cv2.cvtColor(outer_color_hsv, cv2.COLOR_HSV2BGR)
    # Invert the outer color to get the inner color
    inner_color_bgr = cv2.bitwise_not(outer_color_bgr)
    
    # Use the inner pixel color to create a square taking all the space in that color
    # It will be used as a background for the other colors
    background_square = gen_clean_square(int(size), inner_color_bgr[0][0])
    # Create a sqaure in the size of the outer square
    outer_square = gen_clean_square(round_funtion(size * outer_square_ratio), outer_color_bgr[0][0])
    # Create a sqaure in the size of the inner square
    inner_square = gen_clean_square(round_funtion(size * inner_square_ratio), inner_color_bgr[0][0])
    # Paste the outer square into the background square
    start_pos = round_funtion(size * (1 - outer_square_ratio)/2)
    end_pos   = round_funtion(size * ((1 - outer_square_ratio)/2 + outer_square_ratio))
    background_square[start_pos:end_pos, start_pos:end_pos] = outer_square
    # Paste the inner square into the background square
    start_pos = round_funtion(size * (1/2 - (inner_square_ratio)/2))
    end_pos   = round_funtion(size * (1/2 + (inner_square_ratio)/2))
    background_square[start_pos:end_pos, start_pos:end_pos] = inner_square

    return background_square

if __name__ == '__main__':
    # Generate a blue and yellow marker
    blue_and_yellow_fiducial_marker = generate_fiducial_marker(120, 2000)
    # Generate a red and cyan marker
    red_and_cyan_fiducial_marker = generate_fiducial_marker(0, 2000)
    # Generate a green and magenta marker
    green_and_magenta_fiducial_marker = generate_fiducial_marker(50, 2000)

    # Show the markers
    cv2.imshow('blue and yellow', blue_and_yellow_fiducial_marker)
    cv2.imshow('red and cyan', red_and_cyan_fiducial_marker)
    cv2.imshow('green and magenta', green_and_magenta_fiducial_marker)
    cv2.waitKey(0)

    # Save the markers
    cv2.imwrite('blue_and_yellow.png', blue_and_yellow_fiducial_marker)
    cv2.imwrite('red_and_cyan.png', red_and_cyan_fiducial_marker)
    cv2.imwrite('green_and_magenta.png', green_and_magenta_fiducial_marker)
