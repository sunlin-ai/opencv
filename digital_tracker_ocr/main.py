import numpy as np
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
import moviepy.editor as mpy
import cv2
import pytesseract
import sys
import matplotlib


# ------------------------------#
#            Tools
# ------------------------------#
def video_to_wave(video):
    audio_background = mpy.AudioFileClip(video).subclip(1, 60)
    audio_background.write_audiofile('datas/nvh.wav')


# ------------------------------#
#             OCR
# ------------------------------#
def get_digital(image, show_cnts_org=False, show_cnts_filter=False, show_cnts_box=False, show_rotated_img=False):
    if isinstance(image, str):
        img = cv2.imread(image, 1)
    else:
        img = image

    img_org = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 135, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if show_cnts_org:
        cv2.drawContours(img_org, contours, -1, (0, 0, 255), 1)
        plt.imshow(img_org)
        plt.show()

    # # 对轮廓按照面积由大到小排序
    # cnts = sorted(contours, key=cv2.contourArea, reverse=True)[:4]
    img_center = (img.shape[1] // 2, img.shape[0] // 2)
    # cv2.circle(img, img_center, color=(255, 255, 0), radius=2)

    ## 获得数字区域
    cnts_filter = []
    for cnt in contours:
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        else:
            cnt_center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            area = cv2.contourArea(cnt)
            height, width = cv2.minAreaRect(cnt)[1]

            if area > 40 and area < 230 and width < 20:
                center_dis = np.sqrt((img_center[0] - cnt_center[0]) ** 2 + (img_center[1] - cnt_center[1]) ** 2)
                if center_dis < 30:
                    cnts_filter.append(cnt)
                    # cv2.circle(img, cnt_center, color=(255, 0, 0), radius=2)

    if len(cnts_filter) > 1:
        cnts_filter = np.concatenate(cnts_filter, axis=0)

    if show_cnts_filter:
        cv2.drawContours(img, cnts_filter, -1, (0, 0, 255), 1)
        plt.imshow(img)
        plt.show()

    if len(cnts_filter) >= 1:
        cnts_filter = np.array(cnts_filter)
        if len(cnts_filter) == 1:
            x, y, w, h = cv2.boundingRect(cnts_filter[0])
            center = (int(x + w / 2), int(y + h / 2))
            box = [[x - 5, y - 5], [x + w + 5, y - 5], [x + w + 5, y + h + 5], [x - 5, y + h + 5]]
            box = np.int0(box)
            cv2.rectangle(img, box[0], box[2], color=(0, 0, 255), thickness=2)

        else:
            ## 轮廓转换成最小外接矩形
            rect = cv2.minAreaRect(cnts_filter)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            for i in range(4):
                cv2.line(img, tuple(box[i]), tuple(box[(i + 1) % 4]), color=(0, 0, 255), thickness=2)

        if show_cnts_box:
            plt.imshow(img_org)
            plt.show()

        ## 截取数字区域
        img_mask = np.zeros(img.shape[:2], dtype="uint8")
        roi_t = np.expand_dims(box, axis=0)
        cv2.fillPoly(img_mask, roi_t, 255)
        digital_roi = cv2.bitwise_and(img_org, img_org, mask=img_mask)
        ret, binary = cv2.threshold(digital_roi, 135, 255, cv2.THRESH_BINARY_INV)

        ## 旋转图片
        if len(cnts_filter) == 1:
            angle = 56
        else:
            angle = cv2.minAreaRect(cnts_filter)[2]
            M = cv2.moments(cnts_filter)
            center = (int(M["m01"] / M["m00"]), int(M["m10"] / M["m00"]))

        h, w, _ = img.shape
        M = cv2.getRotationMatrix2D(center, angle - 90, 1.0)
        rotated = cv2.warpAffine(binary, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        if show_rotated_img:
            plt.imshow(rotated)
            plt.show()

        ## OCR 识别
        digital = pytesseract.image_to_string(rotated, lang='eng', config='--psm 6 --oem 3 -c '
                                                                          'tessedit_char_whitelist'
                                                                          '=0123456789')
        return digital


# ------------------------------#
#        video tracker
# ------------------------------#
def video_tracker(video, choose_bbox, save_video):
    # 1.创建追踪对象
    # tracker = cv2.TrackerBoosting_create()
    # tracker = cv2.TrackerMIL_create()
    tracker = cv2.TrackerKCF_create()
    # tracker = cv2.TrackerTLD_create()
    # tracker = cv2.TrackerCSRT_create()
    # tracker = cv2.TrackerGOTURN_create()
    # tracker = cv2.TrackerMedianFlow_create()

    video = cv2.VideoCapture(video)

    if not video.isOpened():
        print("Could not open video")
        sys.exit()
    else:
        rate = video.get(5)  # 帧速率
        FrameNumber = video.get(7)  # 视频文件的帧数
        duration = FrameNumber / rate

    ok, frame = video.read()
    if not ok:
        print('Cannot read video file')
        sys.exit()

    # 3.设置跟踪ROI区域
    # 默认跟踪区域
    if choose_bbox == True:
        bbox = cv2.selectROI(frame, False)
    else:
        bbox = choose_bbox

    ok = tracker.init(frame, bbox)
    frame_id = 0
    frame_speed = {}
    # frame_speed[round(0, 2)] = None

    record = [None]

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    fps = video.get(cv2.CAP_PROP_FPS)
    width, height = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter('result/result.mp4', fourcc, fps, (width, height))

    while True:

        time = frame_id / rate

        ok, frame = video.read()
        if not ok:
            break
        timer = cv2.getTickCount()
        ok, bbox = tracker.update(frame)

        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        # 绘制结果
        if ok:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 255, 0), 2, 1)
            cv2.putText(frame, "ROI", (p1[0], p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)

            image = frame[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2]), :]
            # file='datas/output/frame_{}.png'.format(frame_id)
            # cv2.imwrite(file,image)
            digital = get_digital(image, show_cnts_filter=False, show_cnts_box=False, show_rotated_img=False)

            if digital is not None:
                speed = digital.strip()
                if speed.isdigit():
                    record.append(speed)
                else:
                    speed = record[-1]
            else:
                speed = record[-1]

            # res_time_speed[round(time, 2)] = speed
            frame_speed[frame_id] = speed

        else:
            frame_speed[frame_id] = None
            cv2.putText(frame, "Tracking failure detected", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255),
                        2)

        cv2.putText(frame, "time:{:.2f}s, speed:{}".format(time, speed), (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        cv2.imshow("Tracking", frame)

        frame_id += 1

        if save_video:
            out.write(frame)

        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break

    return frame_speed, frame_id


# ------------------------------#
#             FFT
# ------------------------------#
class FFT:
    def __init__(self, file_name, nfft, windowing, overlap):
        self.nfft = nfft
        self.windowing = windowing
        self.overlap = overlap
        self.sample_rate, self.samples = wavfile.read(file_name)
        self.duration = len(self.samples) / self.sample_rate

    def draw_fft(self):
        samples = self.samples.T
        nchannels = samples.shape[0]
        nframes = samples.shape[1]

        yf = np.fft.fft(samples)  # FFT
        bias = (yf[:, 0] / nframes).real
        yf_amplitude = np.abs(yf) * (2.0 / nframes)
        yf_amplitude[:, 0] = bias  # 直流分量(0 Hz处)修正
        self.yf_amplitude = yf_amplitude[:, 0:nframes // 2]  # 有效信息只有一半

        matplotlib.rcParams["axes.unicode_minus"] = False
        self.time = np.arange(0, nframes) * (1.0 / self.sample_rate)
        self.freq = np.arange(0, nframes // 2) * self.sample_rate / nframes  # 实际频率

        for i in range(nchannels):
            plt.subplot(2, nchannels, i + 1)
            plt.plot(self.time, samples[i, :])
            plt.xlabel("time[s]")
            plt.ylabel("signal value")
            plt.grid()
            plt.title("channel%d time signals" % (i + 1))
            plt.subplot(2, nchannels, nchannels + i + 1)
            plt.plot(self.freq, self.yf_amplitude[i, :], "r-")
            plt.xlabel("Frequency[Hz]")
            plt.ylabel("Amplitude")
            plt.grid()
            plt.title("FFT (channel%d frequency signals)" % (i + 1))
        plt.suptitle("NVH FFT", fontsize=14)
        plt.tight_layout()
        plt.show()

    def draw_spectrum(self, frame_speed, video_frame_num):
        windowing = signal.get_window(self.windowing, self.nfft)
        noverlap = self.nfft * self.overlap / 100
        Pxx, freqs, t, im = plt.specgram(self.samples[:, 0], NFFT=self.nfft, Fs=self.sample_rate,
                                         window=windowing, noverlap=noverlap, xextent=(0, self.duration))

        plt.xlabel('Time[s]')
        plt.ylabel('Frequency[Hz]')
        plt.colorbar()
        plt.show()

        img = im.get_array()

        cmap = 'viridis'

        # plt.imshow(img, aspect="auto", cmap=cmap)
        # plt.colorbar()
        # plt.show()

        img = np.rot90(img, -1)

        # plt.imshow(img, aspect="auto", cmap=cmap)
        # plt.colorbar()
        # plt.show()

        gap = img.shape[0] / video_frame_num
        frames = np.arange(0, img.shape[0])
        speeds = [frame_speed[int(item // gap)] for item in frames]
        speeds = [eval(speed) if speed is not None else None for speed in speeds]

        filter = [i for i, a in enumerate(speeds) if a == None]
        speeds = np.delete(speeds, filter, axis=0)
        img = np.delete(img, filter, axis=0)

        temp_img = np.concatenate([img, speeds[:, np.newaxis]], axis=1)

        temp_img = temp_img[np.argsort(temp_img[:, -1])[::-1]]

        img = np.array(temp_img[:, :-1], np.float)

        plt.imshow(img, aspect="auto", cmap=cmap)

        ## adjust x axis
        locs_x, _ = plt.xticks()
        plt.xticks(ticks=np.linspace(0, locs_x[-1], len(locs_x), endpoint=False),
                   labels=[int(item) for item in np.linspace(freqs[0], freqs[-1], len(locs_x))])
        plt.xlabel('Frequency[Hz]')

        ## adjust y axis
        locs_y, _ = plt.yticks()
        plt.yticks(ticks=np.linspace(0, locs_y[-1], len(locs_y), endpoint=False),
                   labels=[int(item) for item in np.linspace(speeds[-1], speeds[0], len(locs_y))])
        plt.ylabel('Speed')
        plt.colorbar()
        plt.show()


if __name__ == '__main__':
    ## video_tracker
    # video = '/home/zhangsong/SUNLIN/project/NVH/datas/video.mp4'
    # # choose_bbox=True
    # choose_bbox = (235, 611, 70, 76)
    # frame_speed, video_frame_num = video_tracker(video, choose_bbox=choose_bbox, save_video=True)

    # print(video_frame_num)
    # print(frame_speed)

    ## FFT
    frame_speed={0: None, 1: None, 2: None, 3: None, 4: None, 5: None, 6: None, 7: '7', 8: '7', 9: '7', 10: '7', 11: '7', 12: '7',
     13: '7', 14: '7', 15: '1', 16: '1', 17: '8', 18: '8', 19: '8', 20: '8', 21: '8', 22: '8', 23: '8', 24: '9',
     25: '9', 26: '9', 27: '9', 28: '9', 29: '9', 30: '9', 31: '9', 32: '9', 33: '9', 34: '9', 35: '9', 36: '9',
     37: '9', 38: '9', 39: '9', 40: '9', 41: '9', 42: '9', 43: '9', 44: '1', 45: '11', 46: '1', 47: '12', 48: '12',
     49: '12', 50: '12', 51: '12', 52: '12', 53: '12', 54: '12', 55: '14', 56: '14', 57: '4', 58: '14', 59: '14',
     60: '14', 61: '1', 62: '15', 63: '15', 64: '15', 65: '15', 66: '15', 67: '15', 68: '15', 69: '7', 70: '7',
     71: '17', 72: '7', 73: '17', 74: '7', 75: '7', 76: '18', 77: '18', 78: '18', 79: '18', 80: '18', 81: '18',
     82: '18', 83: '19', 84: '19', 85: '19', 86: '19', 87: '19', 88: '19', 89: '19', 90: '20', 91: '20', 92: '20',
     93: '20', 94: '20', 95: '20', 96: '20', 97: '20', 98: '21', 99: '21', 100: '21', 101: '21', 102: '21', 103: '21',
     104: '21', 105: '22', 106: '22', 107: '22', 108: '22', 109: '22', 110: '22', 111: '22', 112: '22', 113: '22',
     114: '22', 115: '22', 116: '22', 117: '22', 118: '22', 119: '22', 120: '22', 121: '22', 122: '22', 123: '22',
     124: '22', 125: '22', 126: '22', 127: '22', 128: '22', 129: '22', 130: '22', 131: '22', 132: '22', 133: '22',
     134: '22', 135: '22', 136: '22', 137: '22', 138: '22', 139: '22', 140: '22', 141: '21', 142: '21', 143: '21',
     144: '21', 145: '21', 146: '21', 147: '21', 148: '21', 149: '21', 150: '21', 151: '21', 152: '21', 153: '21',
     154: '21', 155: '21', 156: '20', 157: '20', 158: '20', 159: '20', 160: '20', 161: '20', 162: '20', 163: '19',
     164: '19', 165: '19', 166: '19', 167: '19', 168: '19', 169: '19', 170: '18', 171: '18', 172: '18', 173: '18',
     174: '18', 175: '18', 176: '18', 177: '18', 178: '7', 179: '17', 180: '7', 181: '7', 182: '7', 183: '7', 184: '7',
     185: '15', 186: '15', 187: '15', 188: '15', 189: '15', 190: '15', 191: '15', 192: '14', 193: '14', 194: '14',
     195: '14', 196: '14', 197: '14', 198: '14', 199: '13', 200: '13', 201: '13', 202: '13', 203: '13', 204: '13',
     205: '13', 206: '13', 207: '12', 208: '12', 209: '12', 210: '12', 211: '12', 212: '12', 213: '12', 214: '12',
     215: '10', 216: '10', 217: '10', 218: '10', 219: '10', 220: '10', 221: '9', 222: '9', 223: '9', 224: '9', 225: '9',
     226: '9', 227: '9', 228: '9', 229: '7', 230: '7', 231: '7', 232: '7', 233: '7', 234: '7', 235: '7', 236: '6',
     237: '6', 238: '6', 239: '6', 240: '6', 241: '6', 242: '6', 243: '6', 244: '6', 245: '6', 246: '6', 247: '6',
     248: '6', 249: '6', 250: '5', 251: '5', 252: '5', 253: '5', 254: '5', 255: '5', 256: '5', 257: '5', 258: '5',
     259: '5', 260: '5', 261: '5', 262: '5', 263: '5', 264: '5', 265: '5', 266: '5', 267: '5', 268: '5', 269: '5',
     270: '5', 271: '5', 272: '5', 273: '5', 274: '5', 275: '5', 276: '5', 277: '5', 278: '5', 279: '5', 280: '5',
     281: '5', 282: '5', 283: '5', 284: '5', 285: '5', 286: '5', 287: '5', 288: '5', 289: '5', 290: '5', 291: '5',
     292: '5', 293: '5', 294: '6', 295: '6', 296: '6', 297: '6', 298: '6', 299: '6', 300: '6', 301: '3', 302: '7',
     303: '7', 304: '7', 305: '7', 306: '7', 307: '7', 308: '9', 309: '9', 310: '9', 311: '9', 312: '9', 313: '9',
     314: '9', 315: '9', 316: '10', 317: '10', 318: '10', 319: '10', 320: '10', 321: '10', 322: '10', 323: '10',
     324: '12', 325: '12', 326: '12', 327: '12', 328: '12', 329: '12', 330: '12', 331: '13', 332: '13', 333: '13',
     334: '13', 335: '13', 336: '13', 337: '14', 338: '14', 339: '14', 340: '14', 341: '14', 342: '14', 343: '14',
     344: '14', 345: '15', 346: '15', 347: '15', 348: '15', 349: '15', 350: '15', 351: '15', 352: '15', 353: '16',
     354: '16', 355: '16', 356: '16', 357: '16', 358: '16', 359: '16', 360: '7', 361: '7', 362: '7', 363: '7',
     364: '17', 365: '7', 366: '19', 367: '19', 368: '19', 369: '19', 370: '19', 371: '19', 372: '19', 373: '19',
     374: '19', 375: '19', 376: '19', 377: '19', 378: '19', 379: '19', 380: '19', 381: '19', 382: '21', 383: '21',
     384: '21', 385: '21', 386: '21', 387: '21', 388: '21', 389: '21', 390: '21', 391: '21', 392: '21', 393: '21',
     394: '21', 395: '21', 396: '22', 397: '22', 398: '22', 399: '22', 400: '22', 401: '22', 402: '22', 403: '23',
     404: '23', 405: '23', 406: '23', 407: '23', 408: '23', 409: '23', 410: '23', 411: '24', 412: '24', 413: '24',
     414: '24', 415: '24', 416: '24', 417: '24', 418: '25', 419: '25', 420: '25', 421: '25', 422: '25', 423: '25',
     424: '25', 425: '25', 426: '25', 427: '25', 428: '25', 429: '25', 430: '25', 431: '25', 432: '26', 433: '26',
     434: '26', 435: '26', 436: '26', 437: '26', 438: '26', 439: '26', 440: '26', 441: '26', 442: '26', 443: '26',
     444: '26', 445: '26', 446: '26', 447: '26', 448: '26', 449: '26', 450: '26', 451: '26', 452: '26', 453: '26',
     454: '26', 455: '26', 456: '26', 457: '26', 458: '26', 459: '26', 460: '26', 461: '26', 462: '26', 463: '26',
     464: '26', 465: '26', 466: '26', 467: '26', 468: '26', 469: '26', 470: '26', 471: '26', 472: '26', 473: '26',
     474: '26', 475: '26', 476: '26', 477: '26', 478: '26', 479: '26', 480: '26', 481: '26', 482: '26', 483: '26',
     484: '26', 485: '26', 486: '26', 487: '26', 488: '26', 489: '26', 490: '26', 491: '26', 492: '26', 493: '26',
     494: '26', 495: '26', 496: '26', 497: '26', 498: '26', 499: '25', 500: '25', 501: '25', 502: '25', 503: '25',
     504: '25', 505: '25', 506: '25', 507: '25', 508: '25', 509: '25', 510: '25', 511: '25', 512: '25', 513: '24',
     514: '24', 515: '24', 516: '24', 517: '24', 518: '24', 519: '24', 520: '23', 521: '23', 522: '23', 523: '23',
     524: '23', 525: '23', 526: '23', 527: '22', 528: '22', 529: '22', 530: '22', 531: '22', 532: '22', 533: '22',
     534: '22', 535: '22', 536: '22', 537: '22', 538: '22', 539: '22', 540: '22', 541: '22', 542: '22', 543: '22',
     544: '22', 545: '22', 546: '22', 547: '22', 548: '22', 549: '22', 550: '22', 551: '22', 552: '22', 553: '22',
     554: '22', 555: '22', 556: '22', 557: '22', 558: '22', 559: '22', 560: '22', 561: '22', 562: '22', 563: '22',
     564: '23', 565: '23', 566: '23', 567: '23', 568: '23', 569: '23', 570: '23', 571: '24', 572: '24', 573: '24',
     574: '24', 575: '24', 576: '24', 577: '24', 578: '24', 579: '24', 580: '24', 581: '24', 582: '24', 583: '24',
     584: '24', 585: '26', 586: '26', 587: '26', 588: '26', 589: '26', 590: '26', 591: '26', 592: '26', 593: '26',
     594: '26', 595: '26', 596: '26', 597: '26', 598: '26', 599: '26', 600: '26', 601: '26', 602: '26', 603: '26',
     604: '26', 605: '26', 606: '26', 607: '26', 608: '26', 609: '26', 610: '26', 611: '26', 612: '26', 613: '26',
     614: '26', 615: '26', 616: '26', 617: '26', 618: '26', 619: '26', 620: '26', 621: '26', 622: '25', 623: '25',
     624: '25', 625: '25', 626: '25', 627: '25', 628: '25', 629: '24', 630: '24', 631: '24', 632: '24', 633: '24',
     634: '24', 635: '24', 636: '22', 637: '22', 638: '22', 639: '22', 640: '22', 641: '22', 642: '22', 643: '22',
     644: '21', 645: '21', 646: '21', 647: '21', 648: '21', 649: '21', 650: '21', 651: '21', 652: '21', 653: '21',
     654: '21', 655: '21', 656: '21', 657: '21', 658: '21', 659: '21', 660: '21', 661: '21', 662: '21', 663: '21',
     664: '21', 665: '21', 666: '20', 667: '20', 668: '20', 669: '20', 670: '20', 671: '20', 672: '20', 673: '20',
     674: '20', 675: '20', 676: '20', 677: '20', 678: '20', 679: '20', 680: '21', 681: '21', 682: '21', 683: '21',
     684: '21', 685: '21', 686: '21', 687: '21', 688: '21', 689: '21', 690: '21', 691: '21', 692: '21', 693: '21',
     694: '23', 695: '23', 696: '23', 697: '23', 698: '23', 699: '23', 700: '23', 701: '23', 702: '24', 703: '24',
     704: '24', 705: '24', 706: '24', 707: '24', 708: '24', 709: '25', 710: '25', 711: '25', 712: '25', 713: '25',
     714: '25', 715: '25', 716: '26', 717: '26', 718: '26', 719: '26', 720: '26', 721: '26', 722: '26', 723: '27',
     724: '27', 725: '27', 726: '27', 727: '27', 728: '27', 729: '27', 730: '27', 731: '28', 732: '28', 733: '28',
     734: '28', 735: '28', 736: '28', 737: '28', 738: '28', 739: '28', 740: '28', 741: '28', 742: '28', 743: '28',
     744: '28', 745: '28', 746: '28', 747: '28', 748: '28', 749: '28', 750: '28', 751: '28', 752: '28', 753: '28',
     754: '28', 755: '28', 756: '28', 757: '28', 758: '28', 759: '28', 760: '27', 761: '27', 762: '27', 763: '27',
     764: '27', 765: '27', 766: '27', 767: '27', 768: '27', 769: '27', 770: '27', 771: '27', 772: '27', 773: '27',
     774: '26', 775: '26', 776: '26', 777: '26', 778: '26', 779: '26', 780: '26', 781: '26', 782: '25', 783: '25',
     784: '25', 785: '25', 786: '25', 787: '25', 788: '25', 789: '25', 790: '25', 791: '25', 792: '25', 793: '25',
     794: '25', 795: '25', 796: '25', 797: '24', 798: '24', 799: '24', 800: '24', 801: '24', 802: '24', 803: '23',
     804: '23', 805: '23', 806: '23', 807: '23', 808: '23', 809: '23', 810: '23', 811: '23', 812: '23', 813: '23',
     814: '23', 815: '23', 816: '23', 817: '23', 818: '23', 819: '23', 820: '23', 821: '23', 822: '23', 823: '23',
     824: '23', 825: '23', 826: '23', 827: '23', 828: '23', 829: '23', 830: '23', 831: '23', 832: '23', 833: '23',
     834: '23', 835: '23', 836: '23', 837: '23', 838: '23', 839: '23', 840: '24', 841: '24', 842: '24', 843: '24',
     844: '24', 845: '24', 846: '24', 847: '25', 848: '25', 849: '25', 850: '25', 851: '25', 852: '25', 853: '25',
     854: '25', 855: '25', 856: '27', 857: '27', 858: '27', 859: '27', 860: '27', 861: '27', 862: '28', 863: '28',
     864: '28', 865: '28', 866: '28', 867: '28', 868: '28', 869: '29', 870: '29', 871: '29', 872: '29', 873: '29',
     874: '29', 875: '29', 876: '29', 877: '29', 878: '29', 879: '29', 880: '29', 881: '29', 882: '29', 883: '29',
     884: '28', 885: '28', 886: '28', 887: '28', 888: '28', 889: '28', 890: '28', 891: '28', 892: '28', 893: '28',
     894: '28', 895: '28', 896: '28', 897: '28', 898: '27', 899: '27', 900: '27', 901: '27', 902: '27', 903: '27',
     904: '27', 905: '27', 906: '26', 907: '26', 908: '26', 909: '26', 910: '26', 911: '26', 912: '26', 913: '25',
     914: '25', 915: '25', 916: '25', 917: '25', 918: '25', 919: '25', 920: '24', 921: '24', 922: '24', 923: '24',
     924: '24', 925: '24', 926: '24', 927: '24', 928: '24', 929: '24', 930: '24', 931: '24', 932: '24', 933: '24',
     934: '24', 935: '23', 936: '23', 937: '23', 938: '23', 939: '23', 940: '23', 941: '23', 942: '22', 943: '22',
     944: '22', 945: '22', 946: '22', 947: '22', 948: '22', 949: '22', 950: '22', 951: '22', 952: '22', 953: '22',
     954: '22', 955: '22', 956: '22', 957: '20', 958: '20', 959: '20', 960: '20', 961: '20', 962: '20', 963: '20',
     964: '20', 965: '20', 966: '20', 967: '20', 968: '20', 969: '20', 970: '20', 971: '20', 972: '20', 973: '20',
     974: '20', 975: '20', 976: '20', 977: '20', 978: '20', 979: '20', 980: '20', 981: '20', 982: '20', 983: '20',
     984: '20', 985: '20', 986: '20', 987: '20', 988: '20', 989: '20', 990: '20', 991: '20', 992: '20', 993: '22',
     994: '22', 995: '22', 996: '22', 997: '22', 998: '22', 999: '22', 1000: '22', 1001: '23', 1002: '23', 1003: '23',
     1004: '23', 1005: '23', 1006: None, 1007: None, 1008: None, 1009: None, 1010: None, 1011: None, 1012: None,
     1013: None, 1014: None, 1015: None, 1016: None, 1017: None, 1018: None, 1019: None, 1020: None, 1021: None,
     1022: None, 1023: None, 1024: None, 1025: None, 1026: None, 1027: None, 1028: None, 1029: None, 1030: None,
     1031: None, 1032: None, 1033: None, 1034: None, 1035: None, 1036: None, 1037: None, 1038: None, 1039: None,
     1040: None, 1041: None, 1042: None, 1043: None, 1044: None, 1045: None, 1046: None, 1047: None, 1048: None,
     1049: None, 1050: None, 1051: None, 1052: None, 1053: None, 1054: None, 1055: None, 1056: None, 1057: None,
     1058: None, 1059: None, 1060: None, 1061: None, 1062: None, 1063: None, 1064: None, 1065: None, 1066: None,
     1067: None, 1068: None, 1069: None, 1070: None, 1071: None, 1072: None, 1073: None, 1074: None, 1075: None,
     1076: None, 1077: None, 1078: None, 1079: None, 1080: None, 1081: None, 1082: None, 1083: None, 1084: None,
     1085: None, 1086: None, 1087: None, 1088: None, 1089: None, 1090: None, 1091: None, 1092: None, 1093: None,
     1094: None, 1095: None, 1096: None, 1097: None, 1098: None, 1099: None, 1100: None, 1101: None, 1102: None,
     1103: None, 1104: None, 1105: None, 1106: None, 1107: None, 1108: None, 1109: None, 1110: None, 1111: None,
     1112: None, 1113: None, 1114: None, 1115: None, 1116: None, 1117: None, 1118: None, 1119: None, 1120: None,
     1121: None, 1122: None, 1123: None, 1124: None, 1125: None, 1126: None, 1127: None, 1128: None, 1129: None,
     1130: None, 1131: None, 1132: None, 1133: None, 1134: None, 1135: None, 1136: None, 1137: None, 1138: None,
     1139: None, 1140: None, 1141: None, 1142: None, 1143: None, 1144: None, 1145: None, 1146: None, 1147: None,
     1148: None, 1149: None, 1150: None, 1151: None, 1152: None, 1153: None}

    video_frame_num=1154

    windowing = 'hamming'  # ["hamming", "triang", "blackman", "hann", "bartlett", "flattop", "bohman", "barthann"]
    overlap = 20  # ["10", "20", "30", "40", "50", "60", "70", "80", "90"]
    nfft = 2048  # ["16", "32", "64", "128", "256", "512", "1024", "2048"]
    file_name = 'datas/nvh.wav'
    fft = FFT(file_name, nfft, windowing, overlap)
    fft.draw_fft()
    fft.draw_spectrum(frame_speed, video_frame_num)

    # image = 'datas/output/frame_272.png'  # frame_201.png frame_246.png
    # res,_ = get_digital(image, show_cnts_org=True, show_cnts_filter=True, show_cnts_box=True, show_rotated_img=True)
    # print(res)

    ## OCR
    # image = 'datas/cut3.png'
    # res = get_digital(image)
    # print(res)
