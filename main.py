import cv2
import numpy as np
import pyrealsense2 as rs
import time
import math
import pygame

def detect(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    hsv_min = np.array([0, 150, 0])
    hsv_max = np.array([5, 255, 255])
    mask1 = cv2.inRange(hsv, hsv_min, hsv_max)

    hsv_min = np.array([170, 150, 0])
    hsv_max = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, hsv_min, hsv_max)

    return mask1 + mask2

def analysis_blob(binary_img):
    label = cv2.connectedComponentsWithStats(binary_img)

    n = label[0] - 1
    data = np.delete(label[2], 0, 0)
    center = np.delete(label[3], 0, 0)

    max_index = np.argmax(data[: ,4])

    maxblob = {}

    # maxblob["upper_left"] = (data[:, 0][max_index], data[:, 1][max_index])
    # maxblob["width"] = data[:, 2][max_index]
    # maxblob["height"] = data[:, 3][max_index]
    # maxblob["area"] = data[:, 4][max_index]
    maxblob["center"] = center[max_index]

    return maxblob


def get_depth(depth_frame, x, y):

    depth = depth_frame.get_distance(x, y)
    if depth == 0:
        for y in range(y-2, y+3):
            for x in range(x-2, x+3):
                depth = depth_frame.get_distance(x, y)
                if depth != 0:
                    break
            if depth != 0:
                break
        if depth == 0:
            print("Error : couldn`t get distance !")
            return None


    return x, y, depth

def get_xyz_from_camera(x_px, y_px, depth):
    #get x ratio
    x_dis = x_px - 640
    x_tan = x_dis / 674.4192801798158
    # h_angle = 180/math.pi * math.atan(x_tan)

    #get y ratio
    y_dis = y_px - 360
    y_tan = y_dis / 649.4571918977125
    # v_angle = 180/math.pi * math.atan(y_tan)

    depth_ratio = math.sqrt(x_tan**2 + y_tan**2 + 1)

    #get x(m)
    x_m = depth / depth_ratio * x_tan
    #get y(m)
    y_m = depth / depth_ratio * y_tan

    z_m = depth / depth_ratio

    return x_m, y_m, z_m

def convert_data_xyz(x, y, z, angle):
    #Please put datas of distance from camera to thing.
    cos = math.cos(math.radians(angle))
    sin = math.sin(math.radians(angle))

    y = y * -1
    y_m = -z * sin + y * cos
    z_m = z * cos + y * sin

    # return x, y_m, z_m
    return x, y_m, z_m


pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
# config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

last_time = 0
clock = pygame.time.Clock()
time_txt = open("time.txt", encoding="utf-8", mode="a")


print("Start streaming")
profile = pipeline.start(config)

try:
    while True:

        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()


        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        color_intrinsics = color_frame.profile.as_video_stream_profile().intrinsics

        # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(
            # depth_image, alpha=0.04), cv2.COLORMAP_JET)

        mask = detect(color_image)
        try:
            target = analysis_blob(mask)

            center_x = int(target["center"][0])
            center_y = int(target["center"][1])

            # print("x : ", center_x, "y : ", center_y, end="   ")
            data = get_depth(depth_frame, center_x, center_y)
            if data != None:
                xyz = get_xyz_from_camera(data[0], data[1], data[2])
                print(round(xyz[0], 5),round(xyz[1], 5),round(xyz[2], 5), end="|")

                xyz = convert_data_xyz(xyz[0], xyz[1], xyz[2], 0)
                print(round(xyz[0], 5),round(xyz[1], 5),round(xyz[2], 5))

            cv2.circle(color_image, (data[0], data[1]), 5, (0, 255, 0),
                    thickness=1, lineType=cv2.LINE_AA)
        except Exception as e:
            print("Error : couldn`t get xyz .")

        color_image = cv2.resize(color_image, dsize=(960, 540))
        color_image = cv2.line(color_image, (480, 0), (480, 540), (0,255,0))
        color_image = cv2.line(color_image, (0, 270), (960, 270), (0,255,0))
        cv2.imshow("RealSense  Color Image", color_image)
        # cv2.imshow("RealSense  Mask image", mask)
        # cv2.imshow("RealSense  Depth Image", depth_colormap)

        if cv2.waitKey(1) & 0xff == 27 :
            break

        clock.tick(20)
        print(str(time.time() - last_time), file=time_txt)
        last_time = time.time()


finally:
    time_txt.close()
    pipeline.stop()
    cv2.destroyAllWindows()
