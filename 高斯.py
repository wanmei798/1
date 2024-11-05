import cv2 as cv

def tracking_by_bg_substract() :
    model = cv.createBackgroundSubtractorMOG2()
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    capture = cv.VideoCapture("D:\烟气检测2024年3月\indoor_daytime_smoke.avi")

    while capture.isOpened() :
        ret, frame = capture.read()
        if ret :
            #对图像帧进行MOG2算法检测
            fgmask = model.apply(frame)
            fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, kernel)
            cv.imshow('frame', frame)
            cv.imshow('fgmask', fgmask)

            if cv.waitKey(10) & 0xff == ord('q') :
                break
        else :
            break

    capture.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    tracking_by_bg_substract()
