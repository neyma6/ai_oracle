import os
# Fix macOS 26+ compatibility issue with older tkinter/Tcl setups
os.environ["SYSTEM_VERSION_COMPAT"] = "0"

import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import threading
from datetime import datetime

class VisionApp(tk.Tk):
    def __init__(self, camera_client, motion_detector=None, yolo_vision=None, ai_processor=None):
        super().__init__()
        self.camera_client = camera_client
        self.motion_detector = motion_detector
        self.yolo_vision = yolo_vision
        self.ai_processor = ai_processor
        self.cancel_flag = False
        self.latest_frame = None
        self.latest_snapshot = None
        self.latest_original_frame = None  # Full original frame for LLM
        self.latest_detections = None
        
        self.title("Ai Oracle - Live Camera Stream")
        self.geometry("1600x900")
        
        # ── Top-level horizontal split: Left (video + snapshots) | Right (log) ──
        self.main_pane = tk.PanedWindow(self, orient=tk.HORIZONTAL, sashwidth=4)
        self.main_pane.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # ═══════════════════ LEFT PANEL ═══════════════════
        left_panel = tk.Frame(self.main_pane)
        self.main_pane.add(left_panel, width=1050)
        
        # Image display container
        self.image_label = tk.Label(left_panel)
        self.image_label.pack(pady=10, fill=tk.BOTH, expand=True)
        
        # Status indicator
        self.status_var = tk.StringVar(value="Initializing Stream...")
        self.status_label = tk.Label(left_panel, textvariable=self.status_var, fg="blue", font=("Arial", 14, "bold"))
        self.status_label.pack(pady=5)
        
        # Snapshots display container (for motion detection history)
        self.snapshots_frame = tk.Frame(left_panel)
        self.snapshots_frame.pack(pady=5, fill=tk.X, expand=False, padx=10)
        
        self.snapshot_slots = []
        
        # Create a blank 200x150 image to act as a placeholder
        blank_img = Image.new('RGB', (200, 150), color=(50, 50, 50))
        self.blank_photo = ImageTk.PhotoImage(blank_img)
        
        for i in range(5):
            slot_frame = tk.Frame(self.snapshots_frame)
            slot_frame.pack(side=tk.LEFT, padx=5)
            
            img_lbl = tk.Label(slot_frame, image=self.blank_photo, borderwidth=1, relief="ridge", width=200, height=150)
            img_lbl.image = self.blank_photo
            img_lbl.pack()
            
            txt_lbl = tk.Label(slot_frame, text="", font=("Arial", 9), fg="green", wraplength=200, justify=tk.CENTER, height=2)
            txt_lbl.pack()
            
            self.snapshot_slots.append({'frame': slot_frame, 'image_label': img_lbl, 'text_label': txt_lbl})
            
        self.saved_entries = []
        
        # Close button
        btn_frame = tk.Frame(left_panel)
        btn_frame.pack(pady=10)
        self.btn_close = tk.Button(btn_frame, text="Close Tracking", command=self.close_app, width=15)
        self.btn_close.pack()
        
        # ═══════════════════ RIGHT PANEL ═══════════════════
        right_panel = tk.Frame(self.main_pane)
        self.main_pane.add(right_panel, width=500)
        
        # ─── Classification Log (top half) ───
        log_title = tk.Label(right_panel, text="📋 Classification Log", font=("Arial", 14, "bold"), fg="#333")
        log_title.pack(pady=(10, 5))
        
        # Header row
        header_frame = tk.Frame(right_panel)
        header_frame.pack(fill=tk.X, padx=10)
        tk.Label(header_frame, text="Time", font=("Arial", 11, "bold"), width=12, anchor="w").pack(side=tk.LEFT)
        tk.Label(header_frame, text="Classification", font=("Arial", 11, "bold"), width=20, anchor="w").pack(side=tk.LEFT)
        tk.Label(header_frame, text="Confidence", font=("Arial", 11, "bold"), width=12, anchor="w").pack(side=tk.LEFT)
        
        ttk.Separator(right_panel, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=10, pady=2)
        
        # Scrollable log area
        log_container = tk.Frame(right_panel)
        log_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.log_canvas = tk.Canvas(log_container, highlightthickness=0)
        self.log_scrollbar = ttk.Scrollbar(log_container, orient=tk.VERTICAL, command=self.log_canvas.yview)
        self.log_scrollable = tk.Frame(self.log_canvas)
        
        self.log_scrollable.bind("<Configure>", lambda e: self.log_canvas.configure(scrollregion=self.log_canvas.bbox("all")))
        self.log_canvas_window = self.log_canvas.create_window((0, 0), window=self.log_scrollable, anchor="nw")
        self.log_canvas.configure(yscrollcommand=self.log_scrollbar.set)
        
        # Keep inner frame width in sync with canvas width
        self.log_canvas.bind("<Configure>", lambda e: self.log_canvas.itemconfig(self.log_canvas_window, width=e.width))
        
        self.log_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.log_row_count = 0
        
        # ─── LLM Results (bottom half) ───
        ttk.Separator(right_panel, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=10, pady=5)
        
        llm_title = tk.Label(right_panel, text="🤖 AI Analysis", font=("Arial", 14, "bold"), fg="#333")
        llm_title.pack(pady=(5, 5))
        
        llm_container = tk.Frame(right_panel)
        llm_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        self.llm_canvas = tk.Canvas(llm_container, highlightthickness=0)
        self.llm_scrollbar = ttk.Scrollbar(llm_container, orient=tk.VERTICAL, command=self.llm_canvas.yview)
        self.llm_scrollable = tk.Frame(self.llm_canvas)
        
        self.llm_scrollable.bind("<Configure>", lambda e: self.llm_canvas.configure(scrollregion=self.llm_canvas.bbox("all")))
        self.llm_canvas_window = self.llm_canvas.create_window((0, 0), window=self.llm_scrollable, anchor="nw")
        self.llm_canvas.configure(yscrollcommand=self.llm_scrollbar.set)
        
        # Keep inner frame width in sync with canvas width
        self.llm_canvas.bind("<Configure>", lambda e: self.llm_canvas.itemconfig(self.llm_canvas_window, width=e.width))
        
        self.llm_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.llm_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.llm_row_count = 0
        
        # Handle X button
        self.protocol("WM_DELETE_WINDOW", self.close_app)
        self.focus_force()

        # Start streaming background thread immediately
        threading.Thread(target=self._stream_task, daemon=True).start()
        
        # Start UI render loop
        self.after(30, self._update_ui_frame)

    def _add_log_row(self, time_str, classification, confidence):
        """Add a single row to the classification log."""
        bg = "#2c2c2c" if self.log_row_count % 2 == 0 else "#3a3a3a"
        row_frame = tk.Frame(self.log_scrollable, bg=bg)
        row_frame.pack(fill=tk.X, pady=1)
        
        tk.Label(row_frame, text=time_str, font=("Arial", 10), width=12, anchor="w", bg=bg, fg="#cccccc").pack(side=tk.LEFT)
        tk.Label(row_frame, text=classification, font=("Arial", 10, "bold"), width=20, anchor="w", bg=bg, fg="#5dade2").pack(side=tk.LEFT)
        tk.Label(row_frame, text=confidence, font=("Arial", 10), width=12, anchor="w", bg=bg, fg="#58d68d").pack(side=tk.LEFT)
        
        self.log_row_count += 1
        
        # Auto-scroll to the bottom
        self.log_canvas.update_idletasks()
        self.log_canvas.yview_moveto(1.0)

    def _add_llm_row(self, time_str, result_text):
        """Add a single LLM result row to the AI Analysis panel."""
        bg = "#1e3a5f" if self.llm_row_count % 2 == 0 else "#2a4a6f"
        row_frame = tk.Frame(self.llm_scrollable, bg=bg)
        row_frame.pack(fill=tk.X, pady=2, padx=2)
        
        tk.Label(row_frame, text=time_str, font=("Arial", 10), anchor="nw", bg=bg, fg="#aaccee").grid(row=0, column=0, sticky="nw", padx=(5, 10), pady=3)
        tk.Label(row_frame, text=result_text, font=("Arial", 10), anchor="w", bg=bg, fg="#e0e0e0", wraplength=380, justify=tk.LEFT).grid(row=0, column=1, sticky="w", pady=3)
        row_frame.columnconfigure(1, weight=1)
        
        self.llm_row_count += 1
        
        # Auto-scroll to the bottom
        self.llm_canvas.update_idletasks()
        self.llm_canvas.yview_moveto(1.0)

    def _run_llm_analysis(self, frame, detections, time_str):
        """Run the LLM analysis in a background thread and push results to the UI."""
        try:
            result = self.ai_processor.analyze(frame, detections)
            self.after(0, self._add_llm_row, time_str, result)
        except Exception as e:
            self.after(0, self._add_llm_row, time_str, f"[Error: {e}]")

    def _stream_task(self):
        """Continuously pulls frames from the camera generator in the background."""
        try:
            for frame in self.camera_client.get_stream():
                if self.cancel_flag:
                    break
                self.latest_frame = frame
                
                if self.motion_detector:
                    snapshot = self.motion_detector.detect(frame)
                    if snapshot is not None:
                        detections = []
                        if self.yolo_vision:
                            detections = self.yolo_vision.classify(frame)
                        self.latest_snapshot = snapshot
                        self.latest_original_frame = frame.copy()
                        self.latest_detections = detections
        except Exception as e:
            self.after(0, self.status_var.set, f"Stream Error: {e}")
            self.after(0, lambda: self.status_label.config(fg="red"))

    def _update_ui_frame(self):
        """Pulls the latest frame and renders it to the Tkinter Canvas."""
        if self.cancel_flag:
            return
            
        if self.latest_frame is not None:
            self.status_var.set("🔴 Live Stream Active")
            
            frame_rgb = cv2.cvtColor(self.latest_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img.thumbnail((720, 480))
            
            self.photo = ImageTk.PhotoImage(img)
            self.image_label.config(image=self.photo)
            self.latest_frame = None
            
        # Check if we have a new snapshot from motion detection
        if self.latest_snapshot is not None:
            snap_rgb = cv2.cvtColor(self.latest_snapshot, cv2.COLOR_BGR2RGB)
            snap_img = Image.fromarray(snap_rgb)
            snap_img.thumbnail((200, 150))
            
            new_photo = ImageTk.PhotoImage(snap_img)
            now_str = datetime.now().strftime("%H:%M:%S")
            
            # Filter out unwanted classifications
            SKIP_LABELS = {'fire hydrant'}
            filtered = [d for d in (self.latest_detections or []) if d['label'] not in SKIP_LABELS]
            
            # If everything was filtered out, skip this snapshot entirely
            if not filtered and self.latest_detections:
                self.latest_snapshot = None
                self.latest_original_frame = None
                self.latest_detections = None
                self.after(30, self._update_ui_frame)
                return
            
            # Build label text from detections
            if filtered:
                label_parts = [f"{d['label']} ({d['confidence']:.0%})" for d in filtered]
                label_text = "\n".join(label_parts[:3])
                
                # Add each detection as a separate row in the log
                for d in filtered:
                    self.after(0, self._add_log_row, now_str, d['label'], f"{d['confidence']:.1%}")
            else:
                label_text = "Unknown motion"
                self.after(0, self._add_log_row, now_str, "Unknown motion", "-")
            
            self.saved_entries.insert(0, {'photo': new_photo, 'label_text': label_text})
            
            if len(self.saved_entries) > 5:
                self.saved_entries.pop()
                
            for i, entry in enumerate(self.saved_entries):
                self.snapshot_slots[i]['image_label'].config(image=entry['photo'], width=200, height=150)
                self.snapshot_slots[i]['image_label'].image = entry['photo']
                self.snapshot_slots[i]['text_label'].config(text=entry['label_text'])
            
            # Fire off LLM analysis in background thread
            if self.ai_processor and self.latest_original_frame is not None:
                original_frame = self.latest_original_frame
                detections_for_llm = filtered or []
                threading.Thread(
                    target=self._run_llm_analysis,
                    args=(original_frame, detections_for_llm, now_str),
                    daemon=True
                ).start()
                
            self.latest_snapshot = None
            self.latest_original_frame = None
            self.latest_detections = None
            
        self.after(30, self._update_ui_frame)

    def close_app(self):
        self.cancel_flag = True
        self.destroy()
