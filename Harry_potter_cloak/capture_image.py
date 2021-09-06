import cv2


def capture_image():
    cam = cv2.VideoCapture(0)

    cv2.namedWindow("test")

    print("Hit Space to Save Esc to Quit")

    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        cv2.imshow("test", frame)

        k = cv2.waitKey(1)
        if k % 256 == 32:
            # SPACE pressed
            img_name = "background.jpg"
            cv2.imwrite(img_name, frame)
            print("background image saved !")
            break
        elif k % 256 == 27:
            print("Closing without saving image.")
            break
    cam.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    capture_image()
