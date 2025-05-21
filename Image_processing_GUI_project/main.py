
import sys
import cv2
from PyQt5.QtWidgets import (
    QApplication, QLabel, QVBoxLayout, QHBoxLayout, QGridLayout,
    QMainWindow, QWidget, QTabWidget,QFileDialog
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from PIL import Image as im
import numpy as np
import matplotlib.pyplot as plt
import io
import io
from PIL import Image as im
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image as im, ImageFilter
from scipy.ndimage import convolve
import numpy as np

def plot_image(img, title):
    fig = plt.figure()
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return np.array(im.open(buf))

def plot_histogram(img, title):
    fig = plt.figure()
    plt.hist(img.ravel(), bins=256, range=[0, 255])
    plt.title(title)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return np.array(im.open(buf))

def draw_shift_plots(right_shift, left_shift):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    axes[0, 0].imshow(right_shift, cmap='gray')
    axes[0, 0].set_title('Shift right Image')
    axes[0, 0].axis('off')

    axes[0, 1].hist(right_shift.ravel(), bins=256, range=[0, 255])
    axes[0, 1].set_title('Shift right Histogram')

    axes[1, 0].imshow(left_shift, cmap='gray')
    axes[1, 0].set_title('Shift left Image')
    axes[1, 0].axis('off')

    axes[1, 1].hist(left_shift.ravel(), bins=256, range=[0, 255])
    axes[1, 1].set_title('Shift left Histogram')

    plt.tight_layout()

    buf = io.BytesIO()
    canvas = FigureCanvas(fig)
    canvas.print_png(buf)
    buf.seek(0)

    image = im.open(buf)
    return np.array(image)

def create_histogram_image(img):
    plt.figure(figsize=(4, 3))
    plt.hist(img.ravel(), bins=256, range=[0, 255])
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)

    pil_img = im.open(buf).convert("RGB")
    return np.array(pil_img)

def draw_cv2_histogram(hist):
    hist = hist.flatten()  # تحويل الشكل من (256,1) إلى (256,)
    hist_img = np.full((300, 256), 255, dtype=np.uint8)  # صورة بيضاء لعرض الهستوجرام

    cv2.normalize(hist, hist, alpha=0, beta=300, norm_type=cv2.NORM_MINMAX)

    for x, y in enumerate(hist):
        cv2.line(hist_img, (x, 300), (x, 300 - int(y)), color=0)

    return cv2.cvtColor(hist_img, cv2.COLOR_GRAY2RGB)

def cvimg_to_qpixmap(cv_img):
    if len(cv_img.shape) == 2:
        height, width = cv_img.shape
        q_img = QImage(cv_img.data, width, height, width, QImage.Format_Grayscale8)
    else:
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb_img.shape
        bytes_per_line = channels * width
        q_img = QImage(rgb_img.data, width, height, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(q_img)


def apply_filters(img):
    kernal = (1 / 9) * np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    filters = {
        "Original": img,
        "Gaussian Blur": cv2.GaussianBlur(img, (5, 5), 0),
        "blur": cv2.blur(img, (5, 5)),
        "Median Blur": cv2.medianBlur(img, 5),
        "Bilateral Filter": cv2.bilateralFilter(img, 9, 75, 75),
        "Sharpen": cv2.filter2D(img, -1, np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])),
        "Laplacian": cv2.convertScaleAbs(cv2.Laplacian(img, cv2.CV_64F)),
        "boxFilter": cv2.boxFilter(img, -1, (3, 3), normalize=True),
        "Custam kernal": cv2.filter2D(img, -1, kernal),
        "Histogram Equalization": cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    }
    return filters




def apply_Section3_1(img):
    # Convert NumPy array to a PIL Image for transformations
    pil_img = im.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    Section3 = {
        "Cropped": pil_img.crop((100, 90, 150, 150)),
        "img_resize": pil_img.resize((200, 150)),
        "reduce": pil_img.reduce(4),
        "FLIP_LEFT_RIGHT": pil_img.transpose(im.FLIP_LEFT_RIGHT),
        "FLIP_TOP_BOTTOM": pil_img.transpose(im.FLIP_TOP_BOTTOM),
        "ROTATE_90": pil_img.transpose(im.ROTATE_90),
        "ROTATE_180": pil_img.transpose(im.ROTATE_180),
        "Rotate 70": pil_img.rotate(angle=70, expand=True),  # Now works
        "TRANSPOSE": pil_img.transpose(im.TRANSPOSE),
        "TRANSVERSE": pil_img.transpose(im.TRANSVERSE)
    }

    # Convert PIL Image objects **back** to NumPy arrays for compatibility
    Section3 = {key: np.array(val) for key, val in Section3.items()}

    return Section3


def apply_Section3_2(img):
    # Convert NumPy array to a PIL Image
    pil_img = im.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # Convert to CMYK mode and split channels
    cmyk_img = pil_img.convert("CMYK")
    C, M, Y, K = cmyk_img.split()
    Section3_2 = {
        "img_resize": pil_img.resize((200, 150)),
        "Cyan": np.array(C),
        "Magenta": np.array(M),
        "Yellow": np.array(Y),
        "Black": np.array(K),
        "Marge CMYK":im.merge("CMYK", (C, M, Y, K))
    }
    cvimg_to_qpixmap
    # Convert back to NumPy format for compatibility with OpenCV
    Section3_2 = {key: np.array(val) for key, val in Section3_2.items()}

    return Section3_2


def apply_Section4(img):
    img3 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_arr = np.array(img3)

    r_max = np.max(img_arr)

    # تفادي القسمة على صفر
    if r_max > 0:
        c = 255 / np.log(1 + r_max)
    else:
        c = 1  # تفادي الانفجار العددي

    img_trans = c * np.log(1 + img_arr + 1e-8)

    gamas = [0.1, 0.5, 1.2, 2.2]
    imgs = []
    for gama in gamas:
        c_gama = 255  # أو 255 / (r_max ** gama) إذا أردت ديناميكية
        imgs.append(np.array(c_gama * ((img3 / 255) ** gama), dtype='uint8'))

    pil_img = im.fromarray(img)
    Section4 = {
        "Original": np.array(img),
        "Gray Image CV2": img3,
        "Gray Image PIL": np.array(pil_img.convert(mode='L')),
        "logarithmic Transformation": np.array(img_trans, dtype='uint8'),
        "gama 0.1": imgs[0],
        "gama 0.5": imgs[1],
        "gama 1.2": imgs[2],
        "gama 2.2": imgs[3],
        "Inverse Transformation": 255 - img3
    }
    return Section4


def apply_Section5_1(img):
    img3 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    hist_img = create_histogram_image(img3)

    pil_img = im.fromarray(img)
    img2_g = np.array(pil_img.convert(mode='L'))
    img_h = create_histogram_image(img2_g)

    img_org = img3
    full_sc_str = ((img_org - np.min(img_org)) / (np.max(img_org) - np.min(img_org))) * 255
    full_sc_str = np.clip(full_sc_str, 0, 255).astype(np.uint8)

    gray_hist_cv2 = cv2.calcHist([img3], [0], None, [256], [0, 256])
    gray_hist_img_cv2 = draw_cv2_histogram(gray_hist_cv2)

    stretched_hist_cv2 = cv2.calcHist([full_sc_str], [0], None, [256], [0, 256])
    stretched_hist_img_cv2 = draw_cv2_histogram(stretched_hist_cv2)
    right_shift = np.clip(img + 20, 0, 255).astype('uint8')
    left_shift = np.clip(img - 20, 0, 255).astype('uint8')

    rows, cols = img3.shape
    right_shift = np.roll(img3, shift=50, axis=1)
    left_shift = np.roll(img3, shift=-50, axis=1)

    # رسم الصور والهيستوجرامات كل واحدة على حدة
    right_shift_img = plot_image(right_shift, 'Shift right Image')
    right_shift_hist = plot_histogram(right_shift, 'Shift right Histogram')
    left_shift_img = plot_image(left_shift, 'Shift left Image')
    left_shift_hist = plot_histogram(left_shift, 'Shift left Histogram')
    Section5_1 = {
        "Original": np.array(img),
        "Gray Image CV2": img3,
        "Gray Image Histogram plt": hist_img,
        "Gray Image PIL": img2_g,
        "Gray Image Histogram pil": img_h,
        "Gray Image Histogram Cv2": gray_hist_img_cv2,
        "stretched grayscale image": full_sc_str,
        "stretched grayscale Histogram CV2": stretched_hist_img_cv2,
        "Shift right Image": right_shift_img,
        "Shift right Histogram": right_shift_hist
    }

    return Section5_1
def apply_Section5_2(img):
    img3 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    rows, cols = img3.shape
    right_shift = np.roll(img3, shift=50, axis=1)
    left_shift = np.roll(img3, shift=-50, axis=1)
    left_shift_img = plot_image(left_shift, 'Shift left Image')
    left_shift_hist = plot_histogram(left_shift, 'Shift left Histogram')
    # Equalization
    img_eq = cv2.equalizeHist(img3)

    # رسم الصورة والهيستوجرام كل واحد لوحده
    equalized_img_plot = plot_image(img_eq, 'Equalized Image')
    equalized_hist_plot = plot_histogram(img_eq, 'Equalized Histogram')

    Section5_2 = {
        "Original": np.array(img),
        "Gray Image CV2": img3,
        "Shift left Image": left_shift_img,
        "Shift left Histogram": left_shift_hist,
        "Equalized Image": equalized_img_plot,
        "Equalized Histogram": equalized_hist_plot

    }

    return Section5_2

def apply_Section6(img):
    original = im.fromarray(img)
    gray = np.array(original.convert(mode='L'))
    arr = np.array(gray)

    result_images = {
        "Original": original
    }

    # --- المرحلة 1: فلاتر PIL الجاهزة ---
    filters = {
        "BLUR": ImageFilter.BLUR,
        "DETAIL": ImageFilter.DETAIL,
        "CONTOUR": ImageFilter.CONTOUR,
        "EDGE_ENHANCE": ImageFilter.EDGE_ENHANCE,
        "EDGE_ENHANCE_MORE": ImageFilter.EDGE_ENHANCE_MORE,
        "EMBOSS": ImageFilter.EMBOSS,
        "FIND_EDGES": ImageFilter.FIND_EDGES,
        "SMOOTH": ImageFilter.SMOOTH,
        "SMOOTH_MORE": ImageFilter.SMOOTH_MORE
    }

    for name, f in filters.items():
        filtered = original.filter(f)
        result_images[f"PIL_{name}"] = filtered

    # --- المرحلة 2: فلتر kernel مخصص (Sharpen) ---
    custom_kernel = ImageFilter.Kernel((3, 3), [-1,-1,-1,-1,9,-1,-1,-1,-1])
    sharpened = original.filter(custom_kernel)
    result_images["Custom Kernel Sharpen"] = sharpened

    # --- المرحلة 3: تطبيق Convolution Kernels باستخدام scipy ---
    kernels = {
        "Sobel_X": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
        "Sobel_Y": np.array([[-1,-2,-1], [0,0,0], [1,2,1]]),
        "Gaussian_Blur": (1/16)*np.array([[1,2,1],[2,4,2],[1,2,1]]),
        "Box_Blur": (1/9)*np.ones((3,3))
    }

    for name, kernel in kernels.items():
        result = convolve(arr, kernel)
        result_images[f"Convolve_{name}"] = im.fromarray(result).convert('L')
    return result_images
def apply_Section9(img):
    img3 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    F = np.fft.fft2(img3)
    Fshift = np.fft.fftshift(F)

    M, N = img3.shape
    H_ideal_hp = np.zeros((M, N), dtype=np.float32)
    D0_hp = 50  # Cutoff frequency for high pass filter
    for u in range(M):
        for v in range(N):
            D = np.sqrt((u - M / 2) ** 2 + (v - N / 2) ** 2)
            if (D > D0_hp):
                H_ideal_hp[u, v] = 1
            else:
                H_ideal_hp[u, v] = 0
    H_ideal_hp = np.abs(H_ideal_hp)  # أخذ القيم المطلقة
    H_ideal_hp = cv2.normalize(H_ideal_hp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Apply Ideal High Pass Filter in frequency domain
    Gshift_ideal_hp = Fshift * H_ideal_hp

    # Inverse Fourier Transform
    G_ideal_hp = np.fft.ifftshift(Gshift_ideal_hp)
    g_ideal_hp = np.fft.ifft2(G_ideal_hp)
    g_ideal_hp = np.abs(g_ideal_hp)  # أخذ القيم المطلقة
    g_ideal_hp = cv2.normalize(g_ideal_hp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Gaussian High Pass Filter
    M, N = img3.shape
    H_gaussian_hp = np.zeros((M, N), dtype=np.float32)
    D0_gaussian_hp = 50  # Cutoff frequency for gaussian high pass filter
    for u in range(M):
        for v in range(N):
            D = np.sqrt((u - M / 2) ** 2 + (v - N / 2) ** 2)
            H_gaussian_hp[u, v] = 1 - np.exp(-D ** 2 / (2 * D0_gaussian_hp * D0_gaussian_hp))
    H_gaussian_hp = np.abs(H_gaussian_hp)  # أخذ القيم المطلقة
    H_gaussian_hp = cv2.normalize(H_gaussian_hp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Apply Gaussian High Pass Filter in frequency domain
    Gshift_gaussian_hp = Fshift * H_gaussian_hp

    # Inverse Fourier Transform
    G_gaussian_hp = np.fft.ifftshift(Gshift_gaussian_hp)
    g_gaussian_hp = np.abs(np.fft.ifft2(G_gaussian_hp))

    M, N = img3.shape
    H_ideal_lp = np.zeros((M, N), dtype=np.float32)
    D0_lp = 100  # Cutoff frequency for low pass filter
    for u in range(M):
        for v in range(N):
            D = np.sqrt((u - M / 2) ** 2 + (v - N / 2) ** 2)
            if (D <= D0_lp):
                H_ideal_lp[u, v] = 1
            else:
                H_ideal_lp[u, v] = 0
    H_ideal_lp = np.abs(H_ideal_lp)  # أخذ القيم المطلقة
    H_ideal_lp = cv2.normalize(H_ideal_lp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Apply Ideal Low Pass Filter in frequency domain
    Gshift_ideal_lp = Fshift * H_ideal_lp

    # Inverse Fourier Transform
    G_ideal_lp = np.fft.ifftshift(Gshift_ideal_lp)
    g_ideal_lp = np.abs(np.fft.ifft2(G_ideal_lp))
    g_ideal_lp = np.fft.ifft2(G_ideal_lp)
    g_ideal_lp = np.abs(g_ideal_lp)  # أخذ القيم المطلقة
    g_ideal_lp = cv2.normalize(g_ideal_lp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Gaussian Low Pass Filter
    M, N = img3.shape
    H_gaussian_lp = np.zeros((M, N), dtype=np.float32)
    D0_gaussian_lp = 100  # Cutoff frequency for gaussian low pass filter
    for u in range(M):
        for v in range(N):
            D = np.sqrt((u - M / 2) ** 2 + (v - N / 2) ** 2)
            H_gaussian_lp[u, v] = np.exp(-D ** 2 / (2 * D0_gaussian_lp * D0_gaussian_lp))
    H_gaussian_lp = np.abs(H_gaussian_lp)  # أخذ القيم المطلقة
    H_gaussian_lp = cv2.normalize(H_gaussian_lp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Apply Gaussian Low Pass Filter in frequency domain
    Gshift_gaussian_lp = Fshift * H_gaussian_lp

    # Inverse Fourier Transform
    G_gaussian_lp = np.fft.ifftshift(Gshift_gaussian_lp)
    g_gaussian_lp = np.abs(np.fft.ifft2(G_gaussian_lp))
    g_gaussian_lp = np.abs(g_gaussian_lp)  # أخذ القيم المطلقة
    g_gaussian_lp = cv2.normalize(g_gaussian_lp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    Section9 = {
        "Original": np.array(img),
        "Ideal High Pass Filter (Shifted)": H_ideal_hp,
        "Image after Ideal High Pass Filter": g_ideal_hp,
        "Gaussian High Pass Filter (Shifted)": H_gaussian_hp,
        "Image after Gaussian High Pass Filter": g_gaussian_hp,
        'Ideal Low Pass Filter (Shifted)':H_ideal_lp,
        'Image after Ideal Low Pass Filter':g_ideal_lp,
        'Gaussian Low Pass Filter (Shifted)':H_gaussian_lp,
        'Image after Gaussian Low Pass Filter':g_gaussian_lp
    }
    return Section9

def apply_Section10(img):
    img_noise1 = Add_Gaussian_Noise(img)

    img_noise2 = Add_Salt_Pepper_Noise(img)

    img_noise3 = add_random_noise(img)
    Section10 = {
        "Original": np.array(img),
        "Add_Gaussian_Noise": img_noise1,
        "Add_Salt_Pepper_Noise":img_noise2,
        'add_random_noise':img_noise3
    }
    return Section10

def apply_Section11(img):
    r, g, b = cv2.split(img)
    C = 255 - r
    M = 255 - g
    Y = 255 - b
    CMY = cv2.merge([C, M, Y])
    HSV_img = cv2.cvtColor(CMY, cv2.COLOR_BGR2HSV)
    Pseudo_img = Pseudo_color(img, 5, True)
    Section11 = {
        "Original": np.array(img),
        "Add_Gaussian_Noise": r,
        "Add_Salt_Pepper_Noise":g,
        'add_random_noise':b,
        'add_CMY':CMY,
        'add_HSV':HSV_img,
        "Pseudo_color":Pseudo_img
    }

    return Section11

def Pseudo_color(img,num_regions=5,is_colored = False):
  if is_colored:
    Gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  else:
    Gray_img = img.copy()
  norm_img = cv2.normalize(Gray_img.astype(float),None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
  colored = np.zeros((*Gray_img.shape,3),dtype=np.uint8)
  # Calculate region thresholds
  step = 256 // num_regions
  colors = [
      (255, 0, 0),    # Red
      (0, 255, 0),    # Green
      (0, 0, 255),    # Blue
      (255, 255, 0),  # Yellow
      (255, 0, 255),  # Magenta
      (0, 255, 255)   # Cyan
  ]

  for i in range(num_regions):
      lower = i * step
      upper = (i+1) * step if i < num_regions-1 else 256
      mask = (norm_img >= lower) & (norm_img < upper)
      colored[mask] = colors[i]
  return colored

def Add_Gaussian_Noise(img,mean=0,std=25):
  noise = np.random.normal(mean,std,img.shape).astype(np.uint8)
  img_noise =cv2.add(img ,noise)
  return img_noise

def Add_Salt_Pepper_Noise(img,prob=0.5):
  noise = np.random.rand(img.shape[0],img.shape[1],img.shape[2])
  img_noise = img.copy()
  h, w, c = img_noise.shape
  noisy_pixels = int(h * w * prob)
  #The randint() method returns an integer number selected element from the specified range
  for _ in range(noisy_pixels):
    row, col = np.random.randint(0, h), np.random.randint(0, w)
    if np.random.rand() < 0.5:
      img_noise[row, col] = [0, 0, 0]
    else:
      img_noise[row, col] = [255, 255, 255]

  return img_noise

def add_random_noise(image, intensity=25):
  noisy_image = image.copy()
  noise = np.random.randint(-intensity, intensity + 1, noisy_image.shape)
  noisy_image = np.clip(noisy_image + noise, 0, 255).astype(np.uint8)
  return noisy_image




class FilterTab(QWidget):
    def __init__(self, filters):
        super().__init__()
        layout = QGridLayout()
        self.setLayout(layout)  # Ensure layout is set
        row, col = 0, 0
        for i, (name, img) in enumerate(filters.items()):
            vbox = QVBoxLayout()
            label_img = QLabel()
            pixmap = cvimg_to_qpixmap(img)
            label_img.setPixmap(pixmap.scaled(250, 250, Qt.KeepAspectRatio))
            label_img.setAlignment(Qt.AlignCenter)

            label_name = QLabel(name)
            label_name.setAlignment(Qt.AlignCenter)

            vbox.addWidget(label_img)
            vbox.addWidget(label_name)

            cell = QWidget()
            cell.setLayout(vbox)

            layout.addWidget(cell, row, col)

            col += 1
            if col == 5:
                col = 0
                row += 1

        self.setLayout(layout)


class MainWindow(QMainWindow):
    def __init__(self):  # Corrected __init__
        super().__init__()

        self.setWindowTitle("Image Filter Viewer - Grid View")
        self.setGeometry(100, 100, 1300, 600)

        self.tabs = QTabWidget()
        self.tabs1 = QTabWidget()
        self.setCentralWidget(self.tabs)

        img = self.load_image()
        if img is not None:
            filters = apply_filters(img)
            Section3_1 = apply_Section3_1(img)
            Section3_2 = apply_Section3_2(img)
            Section4 = apply_Section4(img)
            Section5_1 = apply_Section5_1(img)
            Section5_2 = apply_Section5_2(img)
            #Section6 = apply_Section6(img)
            Section9 = apply_Section9(img)
            Section10 = apply_Section10(img)
            Section11 = apply_Section11(img)



            self.tabs.addTab(FilterTab(Section3_1), "Section3_1")
            self.tabs.addTab(FilterTab(Section3_2), "Section3_2")
            self.tabs.addTab(FilterTab(Section4), "Section4")
            self.tabs.addTab(FilterTab(Section5_1), "Section5_1")
            self.tabs.addTab(FilterTab(Section5_2), "Section5_2")
            #self.tabs.addTab(FilterTab(Section6), "Section6")
            self.tabs.addTab(FilterTab(filters), "Filters")
            self.tabs.addTab(FilterTab(Section9), "Section9")
            self.tabs.addTab(FilterTab(Section10), "Section10")
            self.tabs.addTab(FilterTab(Section11), "Section11")
        else:
            print("Image not loaded")
            self.tabs.addTab(QWidget(), "No Image")


    def load_image(self):
        #path = r"C:\Users\user\Downloads\GUI_project\download.png"
        path, _ = QFileDialog.getOpenFileName(self, "Choose Image", "", "Images (*.png *.jpg *.bmp)")
        img = cv2.imread(path)
        if img is None or img.size == 0:  # Extra check
            print(f"Failed to load image from: {path}")
            return None
        return img


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())  # Ensures event loop is running
