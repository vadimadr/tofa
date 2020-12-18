from collections import defaultdict, deque

import cv2

from tofa.colors import rgb
from tofa.image_transforms import rescale


def draw_bbox(image, bbox, color=rgb("green"), line_width=1, text=None):
    bbox = bbox[:4]
    x0, y0, x1, y1 = bbox[:4]
    cv2.rectangle(image, (x0, y0), (x1, y1), rgb(color), line_width)
    if text is not None:

        draw_text(image, text, (x0, y0 - 5), color=color)
    return image


def draw_bboxes(
    image,
    bboxes,
    confidences=None,
    labels=None,
    color=rgb("green"),
    line_width=1,
    caption=None,
):
    for i, bbox in enumerate(bboxes):
        bbox = bbox[:4]
        text = None
        if caption is not None:
            confidence = confidences[i] if confidences is not None else None
            label = labels[i] if labels is not None else None
            text = caption.format(label=label, confidence=confidence)
        draw_bbox(image, bbox, color, line_width, text)


def draw_circle(image, location, color=rgb("blue"), radius=3):
    x, y = location
    cv2.circle(image, (int(x), int(y)), radius, rgb(color))


def draw_line(image, x0, x1, color=rgb("blue"), thickness=2):
    pt1 = (int(x0[0]), int(x0[1]))
    pt2 = (int(x1[0]), int(x1[1]))
    cv2.line(image, pt1, pt2, rgb(color), thickness)


def draw_landmarks(image, landmarks, color=rgb("blue"), numbers=False, caption=None):
    # assume (num_landmarks x 2)
    if caption is None:
        caption = [None] * len(landmarks)
    for i, (landmark, t) in enumerate(zip(landmarks, caption)):
        draw_circle(image, landmark, color, radius=3)
        x, y = int(landmark[0]), int(landmark[1])
        if t is not None:
            draw_text(image, t, (x + 5, y + 5), color=color)
        if numbers and t is None:
            draw_text(image, "{}".format(i), (x + 5, y + 5), color=color)
    return image


def draw_mask(image, mask, color=rgb("blue"), alpha=0.5):
    label_image = image.copy()
    label_image[mask] = rgb(color)

    image_ = image * (1 - alpha) + label_image * alpha
    image[:] = image_.astype(image.dtype)[:]
    return image


def draw_text(
    image,
    text,
    position=(5, 10),
    color=rgb("blue"),
    size=1,
    font=cv2.FONT_HERSHEY_PLAIN,
):
    position_ = int(position[0]), int(position[1])
    cv2.putText(image, str(text), position_, font, size, rgb(color))


def imshow(image, winname="imshow", delay=0, rgb=True, resize=None, keep_aspect=True):
    if rgb:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if resize is not None:
        if isinstance(resize, (int, float)):
            image = rescale(image, scale_factor=resize)
        else:
            image = resize(image, size=resize, keep_aspect=keep_aspect)

    cv2.imshow(winname, image)
    key = cv2.waitKey(delay)
    return key


def imshow_debug(
    image, winname="imshow", wait_space=True, delay=1, maxlen=100, **kwargs
):
    """
    Imshow that supports pausing / rewinding
    space / n - next frame
    p - previous frame
    c - pause / continue execution
    q / esc - exit (raise keyboard interrupt)
    """

    if not hasattr(imshow_debug, "c_key"):
        imshow_debug._paused = False
    if not hasattr(imshow_debug, "frame_history"):
        imshow_debug._frame_history = defaultdict(lambda: deque(maxlen=maxlen))

    imshow_debug._frame_history[winname].append(image)
    ind = -1
    while wait_space and not imshow_debug._paused:
        image = imshow_debug._frame_history[winname][ind]
        key = imshow(image, winname=winname, delay=0, **kwargs)
        # 27 = esc, 32 = space
        if key in (27, ord("q")):
            break
        if key == ord("p"):
            ind = max(ind - 1, -len(imshow_debug._frame_history[winname]))
        if key in (ord("n"), 32):
            ind += 1
            if ind == 0:
                break
        if key == ord("c"):
            imshow_debug._paused = not imshow_debug._paused
    else:
        key = imshow(image, winname=winname, delay=delay, **kwargs)
    if key == ord("c"):
        imshow_debug._paused = not imshow_debug._paused
    if key in (ord("q"), 27):
        cv2.destroyWindow(winname)
        raise KeyboardInterrupt("You pressed esc key.")
    return key