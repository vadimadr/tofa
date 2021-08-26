import sys
from collections import defaultdict, deque
from typing import Union

import cv2

from tofa.colors import rgb
from tofa.image_transforms import rescale


def draw_bbox(image, bbox, color=rgb("green"), line_width=1, text=None):
    x0, y0, x1, y1 = map(_as_int, bbox[:4])
    cv2.rectangle(image, (x0, y0), (x1, y1), rgb(color), line_width)
    if text is not None:
        draw_text(image, text, (x0, y0 - 5), size=1.5, color=color)
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
    pt1 = _as_int(x0[0]), _as_int(x0[1])
    pt2 = _as_int(x1[0]), _as_int(x1[1])
    cv2.line(image, pt1, pt2, rgb(color), thickness)


def draw_landmarks(image, landmarks, color=rgb("blue"), numbers=False, caption=None):
    # assume (num_landmarks x 2)
    if caption is None:
        caption = [None] * len(landmarks)
    for i, (landmark, t) in enumerate(zip(landmarks, caption)):
        draw_circle(image, landmark, color, radius=3)
        x, y = map(_as_int, landmark[:2])
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
    position=(0, 25),
    color=rgb("white"),
    size=2.0,
    font=cv2.FONT_HERSHEY_PLAIN,
    line_type=cv2.LINE_AA,
    thickness=1,
    shadow=False,
):
    x, y = int(position[0]), int(position[1])
    if shadow:
        cv2.putText(
            image,
            str(text),
            (x + 1, y + 1),
            font,
            size,
            rgb("black"),
            thickness + 1,
            line_type,
        )
    cv2.putText(image, str(text), (x, y), font, size, rgb(color), thickness, line_type)


def draw_text_multiline(
    image, lines: Union[str, list], position=(0, 25), vspace=30, *args, **kwargs
):
    if isinstance(lines, str):
        lines = lines.split("\n")
    x, y = position
    for line in lines:
        draw_text(image, line, (x, y), *args, **kwargs)
        y += vspace


def imshow(image, winname="imshow", delay=0, rgb=True, resize=None, keep_aspect=True):
    if rgb:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if resize is not None:
        if isinstance(resize, (int, float)):
            image = rescale(image, scale_factor=resize)
        else:
            image = resize(image, size=resize, keep_aspect=keep_aspect)

    # OpenCV does not define WM_CLASS property.
    # Hence parsable window title is required to recognize imshow windows
    # Use case: add rules for imshow windows in i3wm config
    win_title = f"opencv: {winname}"

    # if window is not opened
    if win_title not in _opened_windows:
        _opened_windows.add(win_title)
        cv2.namedWindow(win_title, cv2.WINDOW_NORMAL)
        if sys.platform == "darwin":
            # workaround to bring new window to foreground in MacOS
            cv2.setWindowProperty(
                win_title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
            )
            cv2.setWindowProperty(win_title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

    cv2.imshow(win_title, image)
    if delay is not None:
        return cv2.waitKey(delay)
    return


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
    if not hasattr(imshow_debug, "_frame_history"):
        # window opened for the first time
        imshow_debug._paused = False
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


LARGE_INT = (1 << 31) - 1


def _as_int(x, lim=LARGE_INT):
    """Workaround integer overflows for out-of-frame drawings."""
    return max(-lim, min(int(x), lim))


_opened_windows = set()
