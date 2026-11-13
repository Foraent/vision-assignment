import argparse
import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise ValueError('视频打不开')
    count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if count == 0:
            print('第一帧：', frame.shape, frame.dtype)
        count += 1
    cap.release()
    print('总帧数：', count)


if __name__ == '__main__':
    main()
