import cv2
import numpy as np
import math

INNER = 0
OUTER = 1
EDGE = 2

X_PIXEL = 0
Y_PIXEL = 1
AREA = 2
COLOR = 3
CONTOUR = 4


class marker_detection():
    def __init__(self, outer_color_hue=120, color_range=60, sv_percentage=0.2,
                 square_ratio=1.2, maximum_size_ratio=0.8, minimum_square_ratio=0.3,
                 minimum_size_inner=2, minimum_size_outer=40, minimum_size_edge=3,
                 max_size_difference=10, max_distance=3, minimum_votes=3,
                 use_backup_detection=False, debug=False):
        '''
        This class detects a specific marker and takes the following parameters:
        outer_color_hue: The hue of the outer square of the marker
        color_range: The range which the filter will search for the color centered on the color, aka +- range/2
        sv_percentage: How large of saturation and value percentage the color filter will accept
        square_ratio: The maximum percentage difference between the width and height of found contours
        maximum_size_ratio: The maximum amount of space a contour is allowed to use, in ratio of the entire image
        minimum_square_ratio: The percentage of allowed empty space from a perfect filled square. e.g. 0.3 means that the contour must at least fill 70% of a tightly fitted rectangle around it
        minimum_size_inner: The minimum sizes of contours allowed to be considered
        minimum_size_outer: 
        minimum_size_edge: 
        max_size_difference: The maximum area ratio difference that the contours must be within to vote on each other
        max_distance: The maximum pixel difference of the centers of the contours must be within to vote on each other
        minimum_votes: How many contours that must agree on the position of the marker
        use_backup_detection: This can be used to find the marker from further away but creates many false positives
        debug: Paint information on self.image
        '''
        # Used to make it easier to display the image
        self.image = None
        self.mask_outer = None
        self.mask_inner = None
        self.mask_edge = None
        
        ### Color filter constants
        # Create a single pixel in HSV spectum
        outer_color_hsv = np.full((1, 1, 3), [outer_color_hue, 255, 255], dtype=np.uint8)
        # Use cv2 to convert from HSV to BGR
        outer_color_bgr = cv2.cvtColor(outer_color_hsv, cv2.COLOR_HSV2BGR)
        # Invert the outer color to get the inner color
        self.inner_color_bgr = cv2.bitwise_not(outer_color_bgr)[0][0]  # Remove the "image" part of the color
        self.outer_color_bgr = outer_color_bgr[0][0]  # Remove the "image" part of the color
 
        self.outer_color_hue = outer_color_hue
        # Calculate the contrast hue
        if outer_color_hue - 180/2 > 0:
            self.inner_color_hue = outer_color_hue - 180/2
        else:
            self.inner_color_hue = outer_color_hue + 180/2
        
        # The range which the filter will search for the color
        # centered on the color, aka +- range/2
        self.color_range = color_range

        # How large of saturation and value percentage the color filter will accept
        self.sv_percentage = sv_percentage


        ### Sorting constants
        # The maximum percentage difference between the width and height
        # of found contours
        self.square_ratio = square_ratio

        # The maximum amount of space a contour is allowed to use
        # In ratio of the entire image
        self.maximum_size_ratio = maximum_size_ratio

        # The percentage of allowed empty space from a perfect filled square
        # e.g. 0.3 means that the contour must at least fill 70% of a tightly fitted rectangle around it
        self.minimum_square_ratio = minimum_square_ratio

        # The minimum sizes of contours allowed to be considered
        self.minimum_size_inner = minimum_size_inner
        self.minimum_size_outer = minimum_size_outer
        self.minimum_size_edge = minimum_size_edge


        ### Voting constraints
        # The maximum area ratio difference that the contours must be within
        # to vote on each other
        self.max_size_difference = max_size_difference

        # The maximum pixel difference of the centers of the contours
        self.max_distance = max_distance
        
        # How many contours that must agree on the position of the marker
        self.minimum_votes = minimum_votes


        ### Non filter related settings
        # The backup detection is a guess where the marker is
        # based only on the blue filter
        # This can be used to find the marker from further away but creates many false positives
        self.use_backup_detection = use_backup_detection

        self.debug = debug


    def extract_marker(self, image):
        '''
        This function takes an image as input and returns the center of a specific marker.
        If 4 * None is returned, then the detection failed to detect any markers
        '''
        # Make a copy of the image to draw information on
        if self.debug:
            self.image = image.copy()

        # Ensure the image exists
        if image is None:
            return None, None, None, None

        # Get the image size
        image_h, image_w, _ = image.shape

        # Find the contours
        contours_blue, contours_yellow, contours_edges = self.get_contours(image)

        # Filter away any wrong contours
        contour_data = self.filter_contours([contours_yellow, contours_blue, contours_edges], image_h, image_w)

        if self.debug:
            for data in contour_data:
                if data[COLOR] == INNER:
                    color = self.inner_color_bgr.tolist()
                elif data[COLOR] == OUTER:
                    color = self.outer_color_bgr.tolist()
                else:
                    color = (255,255,255)
                cv2.drawContours(self.image, [data[CONTOUR]], -1, color=color)

        # No contours detected
        if len(contour_data) == 0:
            return None, None, None, None
        marker_data, backup_detection = self.find_fiducial(contour_data, image_h, image_w)
        
        # No candidate received enough votes
        if marker_data is None:
            return None, None, None, None
        
        # Only the backup detection was received
        if not self.use_backup_detection and backup_detection:
            return None, None, None, None

        _, _, w, h = cv2.boundingRect(marker_data[4])

        return marker_data[X_PIXEL], marker_data[Y_PIXEL], w, h


    def get_contours(self, image):
        '''
        Takes an image as input and finds the contours of three filters. A blue color, a yellow and an edge filter.
        The found contours are the output
        '''

        contours_outer, self.mask_outer = self.get_contours_color(image, OUTER)
        
        contours_inner, self.mask_inner = self.get_contours_color(image, INNER)

        # Get the edge map (contrast map)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(image_gray,100,200)
        # Connect any close edges
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel)
        # Save a copy for debugging
        self.mask_edge = edges
        
        contours_edges, _ = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

        contours_edges = sorted(contours_edges, key=cv2.contourArea)
        contours_edges = contours_edges[::-1]
        return contours_outer, contours_inner, contours_edges


    def get_contours_color(self, image, color):
        '''
        Takes an image and a color argument, which must be either 'OUTER = 1' or 'INNER = 0'
        The function then applies a color filter of the chosen image and returns the mask and contours
        '''
        # Create a completely black version of the image
        # This could be created once beforehand instead if the image is always the same size
        black = np.full((image.shape[0], image.shape[1]), 0, dtype=np.uint8)
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        if color == OUTER:
            # Find all the pixels which are outer
            index_pos = np.where((image_hsv[:,:,0] > self.outer_color_hue - self.color_range/2) &
                                 (image_hsv[:,:,0] < self.outer_color_hue + self.color_range/2) &
                                 (image_hsv[:,:,1] > self.sv_percentage * 255) &
                                 (image_hsv[:,:,2] > self.sv_percentage * 255))
        elif color == INNER:
            # Find all the pixels which are inner
            index_pos = np.where((image_hsv[:,:,0] > self.inner_color_hue - self.color_range/2) &
                                 (image_hsv[:,:,0] < self.inner_color_hue + self.color_range/2) &
                                 (image_hsv[:,:,1] > self.sv_percentage * 255) &
                                 (image_hsv[:,:,2] > self.sv_percentage * 255))
        else:
            raise ValueError('"color" argument for "get_contours_color" must be either "OUTER = 1" or "INNER = 0"')
        
        # Convert the black map to a mask map, by making all the pixels,
        # which were the right color white
        black[index_pos[0], index_pos[1]] = 255
        
        # Create a mask from the borders
        mask = np.bitwise_not(black)
        
        # Create a small erosion to patch up any small holes in the contours
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, kernel)
        
        # Find the contours
        contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        
        # Sort the contours largest to smallets
        contours = sorted(contours, key=cv2.contourArea)
        contours = contours[::-1]
        return contours, mask

    def filter_contours(self, all_three_contours, image_h, image_w):
        '''
        This function applies a series of checks to the contours, and discards any which do not fulfill the criterias.
        The function takes a list of the contours as input, as well as the image size.
        This must be the size of the cropped image, if it is cropped
        If an image is provided, then the accepted contours will be drawn on the image
        Lastly the function returns a list which contains information on the accepted contours
        '''
        contour_data = []
        for i, contours in enumerate(all_three_contours):
            for contour in contours:
                # Checks that the contour is not a triangle
                if len(contour) < 5:
                    break
                # Fit a rectangle to the contour
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                
                (x, y), (w, h), a = rect
                
                # Make sure the rectangle is squareish
                if h * self.square_ratio < w or w * self.square_ratio < h:
                    continue
                
                # Check that the contour does not take up the majority of the image
                if h > image_h * self.maximum_size_ratio or w > image_w * self.maximum_size_ratio:
                    continue
                    
                # Calculate the size of the contour
                area = cv2.contourArea(contour)

                # Filter out very small contours
                if i < EDGE:  # Check if is a color based contour
                    if i == INNER:
                        if area < self.minimum_size_inner:
                            break  # Since the contours are sorted, we can just stop early
                    else:  # OUTER
                        if area < self.minimum_size_outer:
                            break
                else:  # EDGE
                    if h * w < self.minimum_size_edge:
                        break
                
                # Check that the fitted rectangle is mostly filled by the contour
                # This checks that the contour is squareish
                if abs(area - h * w) > area * self.minimum_square_ratio:
                    continue

                # Save the contour
                contour_data.append([x, y, area, i, contour])
        return contour_data

    def find_fiducial(self, contour_data, image_h, image_w):
        '''
        This function looks through the data given by "filter_contours"
        and selects the point where the marker is
        The function takes the points as input, and if an image is given, then the function will
        draw the selected marker point on the image
        This can be quite slow if there is a lot of contours and would benefit from being rewritten
        or made in c++ instead
        '''

        best_candidate = []
        best_candidate_vote = 0
        # Points are sorted inner > outer > edge > large to small contour area
        if len(contour_data) == 0:
            return None, False
        for data in contour_data:
            # Only use the blue conoturs as candidates
            if not data[COLOR] == OUTER:
                continue

            # The candidates always votes for himself
            votes = 1

            ### Inner for loop
            for inner_data in contour_data:
                # Skip points of the same color
                if data[COLOR] == inner_data[COLOR]:
                    continue
                # Ensure the middle of the contours are close to each other
                if math.sqrt((data[X_PIXEL] - inner_data[X_PIXEL])**2 +\
                             (data[Y_PIXEL] - inner_data[Y_PIXEL])**2) > self.max_distance:
                    continue
                # Ensure the difference in contour size is not too large
                if data[AREA] > inner_data[AREA]:
                    max_diff = inner_data[AREA] * self.max_size_difference
                else:
                    max_diff = data[AREA] * self.max_size_difference
                if abs(data[AREA] - inner_data[AREA]) > max_diff:
                    continue
                votes += 1
            
            # Select a new best candidate if it has the most votes
            if votes > best_candidate_vote or len(best_candidate) == 0:
                best_candidate = data
                best_candidate_vote = votes
            # If a candidate has equal votes then prioritize the one that is most in the middle of the image
            elif votes == best_candidate_vote:
                distance_to_middle = np.sqrt((data[X_PIXEL] - image_h/2)**2 + (data[Y_PIXEL] - image_w/2)**2)
                distance_to_middle_best_candidate = np.sqrt((best_candidate[X_PIXEL] - image_h/2)**2 + (best_candidate[Y_PIXEL] - image_w/2)**2)
                if distance_to_middle_best_candidate > distance_to_middle:
                    best_candidate = data
                    best_candidate_vote = votes

            self.image = cv2.putText(self.image, str(votes), (int(data[X_PIXEL]), int(data[Y_PIXEL])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1)

        # If the best candidate have enough votes then return it
        if best_candidate_vote >= self.minimum_votes:
            if self.debug:
                self.image = cv2.circle(self.image, (int(best_candidate[X_PIXEL]), int(best_candidate[Y_PIXEL])),
                                        radius=3, color=(0, 255, 0), thickness=-1)
            return best_candidate, False
        # Otherwise return the best guess
        elif len(best_candidate):
            if self.debug:
                self.image = cv2.circle(self.image, (int(best_candidate[X_PIXEL]), int(best_candidate[Y_PIXEL])),
                                            radius=3, color=(255, 0, 0), thickness=-1)
            return best_candidate, True

        return None, False

# Run this file stand-alone to test the image detection with a plugged in camera
if __name__ == '__main__':
    camera = cv2.VideoCapture(0)
    cam = marker_detection(debug=True)
    while True:
        _, image = camera.read()
        cam.extract_marker(image)
        if cam.image is not None:
            cv2.imshow('test', cam.image)
        else:
            cv2.imshow('test', image)
        key = cv2.waitKey(1)
        if key == 27:
            break
        