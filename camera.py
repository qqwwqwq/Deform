import pyk4a
from matplotlib import pyplot as plt
from pyk4a import Config, PyK4A, ColorResolution
import numpy as np
import pyrosbag
import cv2
import matplotlib
from matplotlib import image
from ctypes import cdll


k4a = PyK4A(Config(color_resolution=ColorResolution.RES_3072P,
                   depth_mode=pyk4a.DepthMode.WFOV_UNBINNED,
                   synchronized_images_only=True,
                   camera_fps=pyk4a.FPS.FPS_5,))
k4a.connect()
# getters and setters directly get and set on device
#k4a.save_calibration_json('/home/hexin/桌面/js' )
#a=json.loads(k4a.get_calibra())
print(k4a.save_calibration_json())
#print(a["CalibrationInformation"]['Cameras'][0])#['Intrinsics']["ModelParameters"])
k4a.exposure_mode_auto = True
k4a.whitebalance_mode_auto = True
k4a.sharpness=4

while 1:
    #img_color = k4a.get_capture(color_only=True)
    img_color,img_depth = k4a.get_capture()  # Would also fetch the depth image
    im = np.asarray(img_depth, np.uint16)

   # for i in im:
      #  for j in i:
          #  if j > 0:
              #  print(j)
    depth_image = cv2.applyColorMap(cv2.convertScaleAbs(img_depth),cv2.COLORMAP_RAINBOW)
   # depth_image = np.asanyarray(img_depth)
    # if np.any(img_color):
   # cv2.imshow('k4a', img_color[:, :, :3])
    cv2.imshow('k4a', img_color)
    oo=np.random.randint(100,800,size=(96,96))
    o=np.asarray(oo,np.uint16)

    depth_image.astype(np.uint16)
   # cv2.imshow('k4a2', im)
    if cv2.waitKey(1) & 0xff == 27:
        cv2.imwrite('/home/hexin/桌面/cool2.png', img_color[:, :, :3])

        #matplotlib.image.imsave('/home/hexin/桌面/deptg.png',im)
        np.save('/home/hexin/桌面/dep.npy',im)
        #cv2.imwrite('/home/hexin/桌面/deptg2.png', depth_image)
        # plt.subplot(121)
        # plt.imshow(img_color[:, :, :3])
        # plt.subplot(122)
        plt.imshow(im)
        plt.show()
        print(im)
        cv2.destroyAllWindows()
        break
    #if cv2.waitKey(1) & 0xff == 32:
   # cv2.imwrite('/home/hexin/桌面/cool.png', img_color)
k4a.disconnect()