import h5py
import numpy as np
import cv2
import matplotlib.pylab as pl
# import utils


def preview(filename):
    log = h5py.File("/home/ubuntu/commaai_org/dataset/log/" + str(filename) + ".h5")
    cam = h5py.File("/home/ubuntu/commaai_org/dataset/camera/" + str(filename) + ".h5")

    frame_stamp = log['cam1_ptr']

    print(len(frame_stamp))


    for i in range(43700,247700 , 10):
        img = cam['X'][frame_stamp[i]]

        angle = log['steering_angle'][i]
        speed = log['speed'][i]
        print('img_file/{}.jpg    {:.4f}    speed = {:.4f}'.format(i, angle, speed))


        # img = img.swapaxes(0, 2).swapaxes(0, 1)
        # img = np.array(img[:, :, :])
        img = np.transpose(img, (1, 2, 0))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # utils.draw_path_on(img, speed, -angle / 10.0, color=(255, 0, 0))

        cv2.imwrite("/home/ubuntu/comma_day_data/img_file/{}.jpg".format(i), img)


        # print(img.shape)
        # img = cv2.flip(img, 180)
        # out_vid.write(img)
        # print(img.shape)

        # resize image
        # img = cv2.resize(img,(2*img.shape[1], 2*img.shape[0]), interpolation = cv2.INTER_CUBIC)
        # edges = cv2.Canny(cv2.cvtColor(img , cv2.COLOR_RGB2GRAY) ,50,150)
        # edges = cv2.GaussianBlur(edges , (3,3) , 0)
        # res = np.hstack((cv2.cvtColor(img , cv2.COLOR_RGB2GRAY) , edges))

        # log details

        # if abs(angle) < 30:
        #     print('{}    Straight    {:.4f}    speed = {:.4f}'.format(i, angle, speed))
        # elif angle < 0:
        #     print('{}    Left        {:.4f}    speed = {:.4f}'.format(i, angle, speed))
        # else:
        #     print('{}    Right       {:.4f}    speed = {:.4f}'.format(i, angle, speed))

        # display

    # out_vid.release()

def main():
    preview('2016-02-02--10-16-58')


if __name__ == '__main__':
    main()
