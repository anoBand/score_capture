import os
import time
from datetime import datetime
import cv2
import mss
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
from PIL import Image
import configparser
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import queue

# --- 설정 파일 및 전역 변수 ---
CONFIG_FILE = 'config.ini'
OUTPUT_FOLDER = "captured_scores"
CAPTURE_INTERVAL_SEC = 1.0  # 캡처 간격 (초)
QUEUE_POLL_MS = 100  # 큐 확인 간격 (ms)


# --- 메인 애플리케이션 클래스 ---
class ScoreCaptureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("악보 자동 캡처")

        # --- 창 크기 및 중앙 정렬 ---
        app_width = 320
        app_height = 220
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x_pos = (screen_width // 2) - (app_width // 2)
        y_pos = (screen_height // 2) - (app_height // 2)
        self.root.geometry(f"{app_width}x{app_height}+{x_pos}+{y_pos}")
        self.root.resizable(False, False)
        self.root.attributes('-topmost', 1)

        self.capture_area = None
        self.is_capturing = False
        self.last_captured_image_gray = None
        self.captured_image_files = []

        # 캡처 영역 선택 시 필요한 변수들
        self.monitor_info = None
        self.virtual_screen_offset_x = 0
        self.virtual_screen_offset_y = 0

        self.capture_thread = None
        self.pdf_thread = None
        self.message_queue = queue.Queue()

        self.create_widgets()
        self.load_config()

        # 앱 시작 시 즉시 모니터 상태 확인
        self._check_monitor_status()

        self.root.after(QUEUE_POLL_MS, self._process_queue)

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_widgets(self):
        frame = ttk.Frame(self.root, padding="10")
        frame.grid(row=0, column=0, sticky=(tk.W + tk.E + tk.N + tk.S))

        ttk.Label(frame, text="민감도 (0.0-1.0):").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.similarity_var = tk.StringVar(value="0.7")
        self.similarity_entry = ttk.Entry(frame, textvariable=self.similarity_var, width=10)
        self.similarity_entry.grid(row=0, column=1, sticky=tk.W)

        ttk.Label(frame, text="시작 딜레이 (초):").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.delay_var = tk.StringVar(value="3")
        self.delay_entry = ttk.Entry(frame, textvariable=self.delay_var, width=10)
        self.delay_entry.grid(row=1, column=1, sticky=tk.W)

        self.select_button = ttk.Button(frame, text="1. 캡처 영역 선택", command=self.select_capture_area)
        self.select_button.grid(row=2, column=0, columnspan=2, sticky=(tk.W + tk.E), pady=10)

        self.start_button = ttk.Button(frame, text="2. 캡처 시작", command=self.start_capture, state=tk.DISABLED)
        self.start_button.grid(row=3, column=0, sticky=(tk.W + tk.E))

        self.stop_button = ttk.Button(frame, text="종료 및 PDF 생성", command=self.stop_capture, state=tk.DISABLED)
        self.stop_button.grid(row=3, column=1, sticky=(tk.W + tk.E))

        self.status_var = tk.StringVar(value="초기화 중...")
        status_label = ttk.Label(self.root, textvariable=self.status_var, padding="10 5", relief=tk.SUNKEN)
        status_label.grid(row=1, column=0, sticky=(tk.W + tk.E + tk.S))

    def update_status(self, message):
        self.status_var.set(message)
        self.root.update_idletasks()

    def _check_monitor_status(self):
        try:
            with mss.mss() as sct:
                monitor_count = len(sct.monitors) - 1  # 0번(전체) 모니터 제외
                if monitor_count > 1:
                    status_msg = f"다중 모니터({monitor_count}개) 감지됨. 영역을 선택하세요."
                else:
                    status_msg = "단일 모니터 감지됨. 영역을 선택하세요."
                self.status_var.set(status_msg)
        except Exception as e:
            print(f"모니터 감지 오류: {e}")
            self.status_var.set("준비 완료. 영역을 선택하세요.")

    def load_config(self):
        config = configparser.ConfigParser()
        if os.path.exists(CONFIG_FILE):
            config.read(CONFIG_FILE)
            threshold = config.get('Settings', 'similarity_threshold', fallback='0.7')
            self.similarity_var.set(threshold)

    def save_config(self):
        config = configparser.ConfigParser()
        config['Settings'] = {'similarity_threshold': self.similarity_var.get()}
        with open(CONFIG_FILE, 'w') as configfile:
            config.write(configfile)

    def select_capture_area(self):
        try:
            # 캡처할 모니터 정보 미리 저장
            with mss.mss() as sct:
                self.monitor_info = sct.monitors[0]  # 전체 가상 화면
                self.virtual_screen_offset_x = self.monitor_info["left"]
                self.virtual_screen_offset_y = self.monitor_info["top"]
        except Exception as e:
            self.update_status(f"모니터 감지 오류: {e}")
            return

        # 3초 카운트다운 시작 (창 숨기지 않음)
        self._selection_countdown(3)

    def _selection_countdown(self, count):
        """영역 선택 전 3초 카운트다운을 수행하여 재생바 숨길 시간 확보"""
        if count > 0:
            self.update_status(f"{count}초 후 화면 캡처... [악보] 창을 클릭하세요!")
            self.select_button.config(state=tk.DISABLED)  # 버튼 비활성화
            self.root.after(1000, self._selection_countdown, count - 1)
        else:
            # 카운트다운 종료, 화면 캡처 실행
            self.update_status("화면 캡처 중...")

            try:
                with mss.mss() as sct:
                    sct_img = sct.grab(self.monitor_info)
                    screenshot = cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGRA2BGR)
            except mss.exception.ScreenShotError as e:
                self.update_status("화면 캡처 실패. 다시 시도하세요.")
                self.select_button.config(state=tk.NORMAL)  # 버튼 활성화
                print(f"캡처 오류: {e}")
                return

            # 캡처 완료 후 버튼 다시 활성화
            self.select_button.config(state=tk.NORMAL)

            # 캡처한 스크린샷으로 영역 선택 창 띄우기
            selector = AreaSelector(screenshot)
            selected_rect = selector.select_area("캡처할 '악보 전체 영역'을 마우스로 드래그하세요.")

            self.root.deiconify()
            self.root.focus_force()  # 메인 창 다시 포커스

            if selected_rect:
                # 선택된 좌표를 가상 화면 절대 좌표로 변환
                self.capture_area = {
                    'top': selected_rect['top'] + self.virtual_screen_offset_y,
                    'left': selected_rect['left'] + self.virtual_screen_offset_x,
                    'width': selected_rect['width'],
                    'height': selected_rect['height']
                }
                self.update_status("영역 선택 완료. 캡처를 시작하세요.")
                self.start_button.config(state=tk.NORMAL)
            else:
                self._check_monitor_status()  # 취소 시 모니터 상태로 복귀

    # --- 캡처 시작 및 스레드 루프 ---

    def start_capture(self):
        try:
            delay = int(self.delay_var.get())
            threshold = float(self.similarity_var.get())
        except ValueError:
            messagebox.showerror("입력 오류", "민감도와 딜레이는 숫자로 입력해야 합니다.")
            return

        self.save_config()
        self.is_capturing = True
        self.start_button.config(state=tk.DISABLED)
        self.select_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)

        self.capture_thread = threading.Thread(
            target=self._capture_thread_target,
            args=(delay, threshold),
            daemon=True
        )
        self.capture_thread.start()

    def stop_capture(self):
        if self.is_capturing:
            self.is_capturing = False
            self.stop_button.config(state=tk.DISABLED)
            self.update_status("캡처 중지 중...")

    def _capture_thread_target(self, delay, threshold):
        try:
            for i in range(delay, 0, -1):
                if not self.is_capturing: break
                self.message_queue.put(('status', f"{i}초 후 캡처를 시작합니다..."))
                time.sleep(1)

            if not self.is_capturing:
                self.message_queue.put(('capture_stopped', None))
                return

            self.message_queue.put(('status', "캡처 진행 중..."))

            with mss.mss() as sct:
                while self.is_capturing:
                    start_time = time.time()

                    sct_img = sct.grab(self.capture_area)
                    current_image_bgr = np.array(sct_img)
                    current_image_gray = cv2.cvtColor(current_image_bgr, cv2.COLOR_BGRA2GRAY)

                    if self.last_captured_image_gray is None:
                        self.save_image(current_image_bgr)
                        self.message_queue.put(('status', "첫 악보 감지! (1번째 저장)"))
                        self.last_captured_image_gray = current_image_gray
                    else:
                        # 재생바 무시를 위해 상위 80%만 비교
                        h, w = current_image_gray.shape
                        # 상위 80% 높이 계산 (하단 20% 무시)
                        crop_h = int(h * 0.80)

                        # 비교할 영역만 잘라내기
                        img_last_cropped = self.last_captured_image_gray[0:crop_h, :]
                        img_current_cropped = current_image_gray[0:crop_h, :]

                        # 잘라낸 이미지로만 유사도 비교
                        score, _ = compare_ssim(img_last_cropped, img_current_cropped, full=True)

                        if score < threshold:
                            self.save_image(current_image_bgr)
                            count = len(self.captured_image_files)
                            self.message_queue.put(('status', f"새 악보 감지! ({count}번째 저장 / 유사도: {score:.2f})"))
                            self.last_captured_image_gray = current_image_gray  # 비교 대상은 원본으로 갱신

                    elapsed = time.time() - start_time
                    sleep_time = CAPTURE_INTERVAL_SEC - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)

        except mss.exception.ScreenShotError as e:
            # 캡처 영역이 (모니터 변경 등으로) 유효하지 않게 된 경우
            self.message_queue.put(('error', f"캡처 영역 오류: 영역이 화면을 벗어난 것 같습니다. 앱을 재시작하고 영역을 다시 선택해주세요. ({e})"))
        except Exception as e:
            self.message_queue.put(('error', f"캡처 오류 발생: {e}"))
        finally:
            self.message_queue.put(('capture_stopped', None))

    def save_image(self, image_bgr):
        if not os.path.exists(OUTPUT_FOLDER):
            os.makedirs(OUTPUT_FOLDER)
        filename = os.path.join(OUTPUT_FOLDER, f"score_page_{len(self.captured_image_files) + 1:03d}.png")
        cv2.imwrite(filename, cv2.cvtColor(image_bgr, cv2.COLOR_BGRA2BGR))
        self.captured_image_files.append(filename)

    def _pdf_creation_thread_target(self):
        try:
            self.message_queue.put(('status', "PDF 생성 중... 잠시만 기다려주세요."))

            if not self.captured_image_files:
                self.message_queue.put(('pdf_done', (None, "캡처된 이미지가 없어 PDF를 생성하지 않았습니다.")))
                return

            image_list = [Image.open(f) for f in self.captured_image_files]

            # PDF 파일명에 타임스탬프 추가
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            pdf_filename = f"score-{timestamp}.pdf"

            image_list[0].save(
                pdf_filename, "PDF", resolution=100.0,
                save_all=True, append_images=image_list[1:]
            )

            self.message_queue.put(('pdf_done', (pdf_filename, f"'{pdf_filename}' 파일이 성공적으로 생성되었습니다.")))

        except Exception as e:
            self.message_queue.put(('error', f"PDF 생성 오류: {e}"))

    def reset_state(self):
        self.last_captured_image_gray = None
        self.captured_image_files = []
        self.is_capturing = False
        self.start_button.config(state=tk.NORMAL if self.capture_area else tk.DISABLED)
        self.select_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        # 상태 초기화 시 모니터 상태로 복귀
        self._check_monitor_status()

    def _process_queue(self):
        try:
            message_type, value = self.message_queue.get_nowait()

            if message_type == 'status':
                self.update_status(value)
            elif message_type == 'error':
                messagebox.showerror("오류 발생", value)
                self.reset_state()
            elif message_type == 'capture_stopped':
                self.pdf_thread = threading.Thread(
                    target=self._pdf_creation_thread_target,
                    daemon=True
                )
                self.pdf_thread.start()
            elif message_type == 'pdf_done':
                pdf_filename, message = value
                if pdf_filename:
                    messagebox.showinfo("성공", message)
                else:
                    messagebox.showwarning("알림", message)
                self.reset_state()

        except queue.Empty:
            pass
        finally:
            self.root.after(QUEUE_POLL_MS, self._process_queue)

    def on_closing(self):
        self.save_config()
        self.is_capturing = False
        self.root.destroy()


# --- 영역 선택 클래스 (AreaSelector) ---
class AreaSelector:
    def __init__(self, screen_shot):
        self.image = screen_shot
        self.point1 = None
        self.point2 = None
        self.rect_done = False

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.point1 = (x, y)
            self.rect_done = False
        elif event == cv2.EVENT_MOUSEMOVE and self.point1:
            img_copy = self.image.copy()
            cv2.rectangle(img_copy, self.point1, (x, y), (0, 255, 0), 2)
            cv2.imshow("Area Selector", img_copy)
        elif event == cv2.EVENT_LBUTTONUP:
            self.point2 = (x, y)
            self.rect_done = True

    def select_area(self, instructions):
        self.point1 = None
        self.point2 = None
        self.rect_done = False
        cv2.namedWindow("Area Selector", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Area Selector", self.mouse_callback)
        img_with_text = self.image.copy()
        cv2.putText(img_with_text, instructions, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Area Selector", img_with_text)

        # AreaSelector 창이 모든 창 위에 오도록 시도
        cv2.setWindowProperty("Area Selector", cv2.WND_PROP_TOPMOST, 1)

        while not self.rect_done:
            if cv2.waitKey(1) & 0xFF == 27:
                cv2.destroyAllWindows()
                return None

        cv2.destroyAllWindows()

        left = min(self.point1[0], self.point2[0])
        top = min(self.point1[1], self.point2[1])
        width = abs(self.point1[0] - self.point2[0])
        height = abs(self.point1[1] - self.point2[1])

        if width == 0 or height == 0:
            return None

        return {'top': top, 'left': left, 'width': width, 'height': height}


# --- 프로그램 실행 ---
if __name__ == "__main__":
    if os.name == 'nt':
        try:
            from ctypes import windll

            windll.shcore.SetProcessDpiAwareness(1)
        except Exception as e:
            print(f"DPI Awareness 설정 실패: {e}")

    root = tk.Tk()
    app = ScoreCaptureApp(root)
    root.mainloop()