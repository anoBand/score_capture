import os
import sys
import time
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
        # --- 기본 설정 ---
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
        self.root.resizable(False, False)  # 창 크기 조절 불가

        # --- (개선) 항상 위에 떠있는 창으로 설정 ---
        self.root.attributes('-topmost', 1)

        # --- 상태 변수 ---
        self.capture_area = None
        self.is_capturing = False
        self.last_captured_image_gray = None
        self.captured_image_files = []

        # --- 스레드 및 큐 관련 ---
        self.capture_thread = None
        self.pdf_thread = None
        self.message_queue = queue.Queue()  # 스레드 -> GUI 통신용

        # --- UI 위젯 생성 ---
        self.create_widgets()
        self.load_config()

        # --- 큐 폴링 시작 ---
        self.root.after(QUEUE_POLL_MS, self._process_queue)

        # --- 종료 시 처리 ---
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_widgets(self):
        # --- 프레임 설정 ---
        frame = ttk.Frame(self.root, padding="10")
        frame.grid(row=0, column=0, sticky=(tk.W + tk.E + tk.N + tk.S))  # sticky 수정

        # --- 설정 값 입력 UI ---
        ttk.Label(frame, text="민감도 (0.0-1.0):").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.similarity_var = tk.StringVar(value="0.7")  # 기본값 0.7로 변경
        self.similarity_entry = ttk.Entry(frame, textvariable=self.similarity_var, width=10)
        self.similarity_entry.grid(row=0, column=1, sticky=tk.W)

        ttk.Label(frame, text="시작 딜레이 (초):").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.delay_var = tk.StringVar(value="3")
        self.delay_entry = ttk.Entry(frame, textvariable=self.delay_var, width=10)
        self.delay_entry.grid(row=1, column=1, sticky=tk.W)

        # --- 버튼 UI ---
        self.select_button = ttk.Button(frame, text="1. 캡처 영역 선택", command=self.select_capture_area)
        self.select_button.grid(row=2, column=0, columnspan=2, sticky=(tk.W + tk.E), pady=10)  # sticky 수정

        self.start_button = ttk.Button(frame, text="2. 캡처 시작", command=self.start_capture, state=tk.DISABLED)
        self.start_button.grid(row=3, column=0, sticky=(tk.W + tk.E))  # sticky 수정

        self.stop_button = ttk.Button(frame, text="종료 및 PDF 생성", command=self.stop_capture, state=tk.DISABLED)
        self.stop_button.grid(row=3, column=1, sticky=(tk.W + tk.E))  # sticky 수정

        # --- 상태 표시줄 ---
        self.status_var = tk.StringVar(value="영역을 먼저 선택해주세요.")
        status_label = ttk.Label(self.root, textvariable=self.status_var, padding="10 5", relief=tk.SUNKEN)
        status_label.grid(row=1, column=0, sticky=(tk.W + tk.E + tk.S))  # sticky 수정

    def update_status(self, message):
        self.status_var.set(message)
        self.root.update_idletasks()

    def load_config(self):
        config = configparser.ConfigParser()
        if os.path.exists(CONFIG_FILE):
            config.read(CONFIG_FILE)
            threshold = config.get('Settings', 'similarity_threshold', fallback='0.7')  # fallback 0.7로
            self.similarity_var.set(threshold)

    def save_config(self):
        config = configparser.ConfigParser()
        config['Settings'] = {'similarity_threshold': self.similarity_var.get()}
        with open(CONFIG_FILE, 'w') as configfile:
            config.write(configfile)

    def select_capture_area(self):
        try:
            with mss.mss() as sct:
                # 1. (개선) 모니터 정보 감지 및 콘솔(print)이 아닌 상태바(update_status)에 표시
                monitor_count = len(sct.monitors) - 1  # 0번(전체) 모니터 제외
                if monitor_count > 1:
                    status_msg = f"다중 모니터({monitor_count}개) 감지됨. 전체 화면에서 선택하세요..."
                else:
                    status_msg = "단일 모니터 감지됨. 화면에서 선택하세요..."
                self.update_status(status_msg)
                self.root.update_idletasks()  # 상태바 메시지 즉시 업데이트

                # 2. (개선) withdraw() 대신 '투명화'로 깜빡임 제거
                self.root.attributes('-alpha', 0.0)
                # OS가 창을 투명하게 처리할 시간을 줌 (0.1~0.2초)
                time.sleep(0.6)

                # 3. 전체 가상 화면(모든 모니터) 캡처
                sct_img = sct.grab(sct.monitors[0])
                screenshot = cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGRA2BGR)

                # 4. (개선) 캡처 직후 즉시 창 복원
                self.root.attributes('-alpha', 1.0)

                # 가상 화면의 시작점(좌표 보정용)
                virtual_screen_offset_x = sct.monitors[0]["left"]
                virtual_screen_offset_y = sct.monitors[0]["top"]

        except mss.exception.ScreenShotError as e:
            self.update_status("화면 캡처 실패. 다시 시도하세요.")
            self.root.attributes('-alpha', 1.0)  # 오류 발생 시 창 복원
            print(f"캡처 오류: {e}")
            return

        # 5. 캡처한 스크린샷으로 영역 선택 창 띄우기
        selector = AreaSelector(screenshot)
        selected_rect = selector.select_area("캡처할 '악보 전체 영역'을 마우스로 드래그하세요.")

        # 6. 메인 창 포커스 복원
        self.root.deiconify()  # 최소화되었을 경우를 대비
        self.root.focus_force()  # 창을 맨 앞으로 가져옴

        if selected_rect:
            # 선택된 좌표를 mss가 사용할 가상 화면 절대 좌표로 변환
            self.capture_area = {
                'top': selected_rect['top'] + virtual_screen_offset_y,
                'left': selected_rect['left'] + virtual_screen_offset_x,
                'width': selected_rect['width'],
                'height': selected_rect['height']
            }
            self.update_status("영역 선택 완료. 캡처를 시작하세요.")
            self.start_button.config(state=tk.NORMAL)
        else:
            self.update_status("영역 선택이 취소되었습니다.")

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
                        # (개선) 첫 이미지 저장 시 카운트 표시
                        self.save_image(current_image_bgr)
                        self.message_queue.put(('status', "첫 악보 감지! (1번째 저장)"))
                        self.last_captured_image_gray = current_image_gray  # 저장 후 비교 이미지 할당
                    else:
                        score, _ = compare_ssim(self.last_captured_image_gray, current_image_gray, full=True)
                        if score < threshold:
                            # (개선) 새 이미지 저장 시 카운트 표시
                            self.save_image(current_image_bgr)
                            count = len(self.captured_image_files)
                            self.message_queue.put(('status', f"새 악보 감지! ({count}번째 저장 / 유사도: {score:.2f})"))
                            self.last_captured_image_gray = current_image_gray  # 저장 후 비교 이미지 할당

                    elapsed = time.time() - start_time
                    sleep_time = CAPTURE_INTERVAL_SEC - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)

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

            pdf_filename = "final_sheet_music_stitched.pdf"

            image_list[0].save(
                pdf_filename,
                "PDF",
                resolution=100.0,
                save_all=True,
                append_images=image_list[1:]
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
        self.update_status("준비 완료. 영역을 선택하거나 캡처를 시작하세요.")

    def _process_queue(self):
        try:
            message_type, value = self.message_queue.get_nowait()

            if message_type == 'status':
                self.update_status(value)

            elif message_type == 'error':
                messagebox.showerror("오류 발생", value)
                self.reset_state()

            elif message_type == 'capture_stopped':
                # 캡처 스레드 종료 확인 후 PDF 생성 스레드 시작
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

        while not self.rect_done:
            # ESC 키로 취소
            if cv2.waitKey(1) & 0xFF == 27:
                cv2.destroyAllWindows()
                return None

        cv2.destroyAllWindows()

        left = min(self.point1[0], self.point2[0])
        top = min(self.point1[1], self.point2[1])
        width = abs(self.point1[0] - self.point2[0])
        height = abs(self.point1[1] - self.point2[1])

        # 유효하지 않은 영역(크기가 0) 선택 방지
        if width == 0 or height == 0:
            return None

        return {'top': top, 'left': left, 'width': width, 'height': height}


# --- 프로그램 실행 ---
if __name__ == "__main__":
    # Windows에서 DPI 인식 관련 문제 해결 (선명한 GUI)
    if os.name == 'nt':
        try:
            from ctypes import windll

            windll.shcore.SetProcessDpiAwareness(1)
        except Exception as e:
            print(f"DPI Awareness 설정 실패: {e}")

    root = tk.Tk()
    app = ScoreCaptureApp(root)
    root.mainloop()