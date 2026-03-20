import os
# Fix macOS 26+ compatibility issue with older tkinter/Tcl setups
os.environ["SYSTEM_VERSION_COMPAT"] = "0"

import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import threading
import time
from datetime import datetime
import queue

class VisionApp(tk.Tk):
    def __init__(self, camera_client, motion_detector=None, yolo_vision=None, ai_processor=None, db_client=None):
        super().__init__()
        self.camera_client = camera_client
        self.motion_detector = motion_detector
        self.yolo_vision = yolo_vision
        self.ai_processor = ai_processor
        self.db_client = db_client
        self.cancel_flag = False
        self._start_time = time.time()
        self._warmup_seconds = 10
        self.latest_frame = None
        self.latest_snapshot = None
        self.latest_original_frame = None  # Full original frame for LLM
        self.latest_detections = None
        self.llm_queue = queue.Queue()
        self.last_llm_labels = None
        self.last_motion_seen_time = 0



        
        
        self.title("Ai Oracle - Live Camera Stream")
        self.geometry("1600x900")
        
        # ── Top-level horizontal split: Left (video + snapshots) | Right (log) ──
        self.main_pane = tk.PanedWindow(self, orient=tk.HORIZONTAL, sashwidth=4)
        self.main_pane.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # ═══════════════════ LEFT PANEL ═══════════════════
        left_panel = tk.Frame(self.main_pane)
        self.main_pane.add(left_panel, width=1050)
        
        # Image display container
        self.video_images_frame = tk.Frame(left_panel)
        self.video_images_frame.pack(pady=10, fill=tk.BOTH, expand=True)
        
        self.video_frame = tk.Frame(self.video_images_frame)
        self.video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.image_label = tk.Label(self.video_frame)
        self.image_label.pack(pady=5, fill=tk.BOTH, expand=True)
        
        self.selected_image_frame = tk.Frame(self.video_images_frame, width=340, bg="#222")
        self.selected_image_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10)
        self.selected_image_frame.pack_propagate(False)
        
        self.selected_image_label = tk.Label(self.selected_image_frame, text="Select log\nto view image", fg="gray", bg="#222")
        self.selected_image_label.pack(pady=5, fill=tk.BOTH, expand=True)
        
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
        
        # ─── RAG Chat Section ───
        chat_outer = tk.Frame(left_panel, bg="#1a1a2e", relief="ridge", bd=1)
        chat_outer.pack(pady=(5, 0), fill=tk.BOTH, expand=True, padx=10)

        chat_title = tk.Label(chat_outer, text="💬 Ask about past events (RAG)", font=("Arial", 12, "bold"),
                              fg="#7eb8f7", bg="#1a1a2e")
        chat_title.pack(pady=(6, 2))

        # Scrollable chat transcript area
        chat_scroll_frame = tk.Frame(chat_outer, bg="#1a1a2e")
        chat_scroll_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 4))

        self.chat_canvas = tk.Canvas(chat_scroll_frame, bg="#12122a", highlightthickness=0, height=160)
        chat_vsb = ttk.Scrollbar(chat_scroll_frame, orient=tk.VERTICAL, command=self.chat_canvas.yview)
        self.chat_scrollable = tk.Frame(self.chat_canvas, bg="#12122a")
        self.chat_scrollable.bind(
            "<Configure>",
            lambda e: self.chat_canvas.configure(scrollregion=self.chat_canvas.bbox("all"))
        )
        self.chat_canvas_window = self.chat_canvas.create_window((0, 0), window=self.chat_scrollable, anchor="nw")
        self.chat_canvas.configure(yscrollcommand=chat_vsb.set)
        self.chat_canvas.bind("<Configure>",
            lambda e: self.chat_canvas.itemconfig(self.chat_canvas_window, width=e.width))
        self.chat_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        chat_vsb.pack(side=tk.RIGHT, fill=tk.Y)

        # Input row
        input_row = tk.Frame(chat_outer, bg="#1a1a2e")
        input_row.pack(fill=tk.X, padx=6, pady=(0, 6))

        self.chat_entry = tk.Entry(input_row, font=("Arial", 11), bg="#2b2b4b", fg="white",
                                   insertbackground="white", relief="flat", bd=4)
        self.chat_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=4)
        self.chat_entry.bind("<Return>", lambda e: self._send_chat_prompt())

        send_btn = tk.Button(input_row, text="Send", font=("Arial", 10, "bold"),
                             bg="#3a7bd5", fg="white", relief="flat", padx=10,
                             activebackground="#2560b0", cursor="hand2",
                             command=self._send_chat_prompt)
        send_btn.pack(side=tk.LEFT, padx=(6, 0))

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

        # Load historical logs from today & yesterday
        self._load_history()

        # Start streaming background thread immediately
        threading.Thread(target=self._stream_task, daemon=True).start()
        
        # Start UI render loop
        self.after(30, self._update_ui_frame)
        
        # Start background queue processor
        threading.Thread(target=self._llm_worker, daemon=True).start()

    def _llm_worker(self):
        try:
            from ai_oracle.rag_image_converter import RagImageConverter
            # Initialize inside the worker thread so CLIP downloads don't block the UI
            self.rag_converter = RagImageConverter()
        except Exception as e:
            self.rag_converter = None
            print(f"RAG initialization error: {e}")

        while not self.cancel_flag:
            try:
                frame, detections, time_str = self.llm_queue.get(timeout=1.0)
                if self.cancel_flag:
                    break
                self._run_llm_analysis(frame, detections, time_str)
                self.llm_queue.task_done()
                
                # Throttle processing to allow the computer (fans) to cool down between LLM runs
                time.sleep(10.0)

            except queue.Empty:
                continue
            except Exception as e:
                print(f"LLM Worker error: {e}")

    def _show_selected_image(self, image_bytes):
        if not image_bytes:
            self.selected_image_label.config(image='', text="No image available")
            return
        import io
        try:
            pil_image = Image.open(io.BytesIO(image_bytes))
            photo = ImageTk.PhotoImage(pil_image)
            self.selected_image_label.config(image=photo, text="")
            self.selected_image_label.image = photo
        except Exception as e:
            self.selected_image_label.config(image='', text=f"Error loading image: {e}")

    def _add_log_row(self, time_str, classification, confidence, image_bytes=None, log_id=None, persist=True):
        """Add a single row to the classification log."""
        # Persist to database
        if persist and self.db_client:
            try:
                event_date = datetime.now().date().isoformat()
                log_id = self.db_client.save_event_log(time_str, classification, confidence, image_bytes, event_date=event_date)
            except Exception as e:
                print(f"[DB] Error saving event log: {e}")

        bg = "#2c2c2c" if self.log_row_count % 2 == 0 else "#3a3a3a"
        row_frame = tk.Frame(self.log_scrollable, bg=bg)
        row_frame.pack(fill=tk.X, pady=1)
        
        def on_click(event, current_id=log_id):
            if current_id is not None and self.db_client:
                img_data = self.db_client.get_event_image(current_id)
                self._show_selected_image(bytes(img_data) if img_data else None)
            else:
                self._show_selected_image(None)
            
        row_frame.bind("<Button-1>", on_click)
        
        lbl1 = tk.Label(row_frame, text=time_str, font=("Arial", 10), width=12, anchor="w", bg=bg, fg="#cccccc")
        lbl1.pack(side=tk.LEFT)
        lbl1.bind("<Button-1>", on_click)
        
        lbl2 = tk.Label(row_frame, text=classification, font=("Arial", 10, "bold"), width=20, anchor="w", bg=bg, fg="#5dade2")
        lbl2.pack(side=tk.LEFT)
        lbl2.bind("<Button-1>", on_click)
        
        lbl3 = tk.Label(row_frame, text=confidence, font=("Arial", 10), width=12, anchor="w", bg=bg, fg="#58d68d")
        lbl3.pack(side=tk.LEFT)
        lbl3.bind("<Button-1>", on_click)
        
        self.log_row_count += 1
        
        # Auto-scroll to the bottom
        self.log_canvas.update_idletasks()
        self.log_canvas.yview_moveto(1.0)

    def _add_llm_row(self, time_str, result_text, image_bytes=None, log_id=None, persist=True):
        """Add a single LLM result row to the AI Analysis panel."""
        # Persist to database
        if persist and self.db_client:
            try:
                event_date = datetime.now().date().isoformat()
                log_id = self.db_client.save_ai_analysis(time_str, result_text, image_bytes, event_date=event_date)
            except Exception as e:
                print(f"[DB] Error saving AI analysis: {e}")

        bg = "#1e3a5f" if self.llm_row_count % 2 == 0 else "#2a4a6f"
        row_frame = tk.Frame(self.llm_scrollable, bg=bg)
        row_frame.pack(fill=tk.X, pady=2, padx=2)
        
        def on_click(event, current_id=log_id):
            if current_id is not None and self.db_client:
                img_data = self.db_client.get_ai_image(current_id)
                self._show_selected_image(bytes(img_data) if img_data else None)
            else:
                self._show_selected_image(None)
            
        row_frame.bind("<Button-1>", on_click)
        
        lbl1 = tk.Label(row_frame, text=time_str, font=("Arial", 10), anchor="nw", bg=bg, fg="#aaccee")
        lbl1.grid(row=0, column=0, sticky="nw", padx=(5, 10), pady=3)
        lbl1.bind("<Button-1>", on_click)
        
        lbl2 = tk.Label(row_frame, text=result_text, font=("Arial", 10), anchor="w", bg=bg, fg="#e0e0e0", wraplength=380, justify=tk.LEFT)
        lbl2.grid(row=0, column=1, sticky="w", pady=3)
        lbl2.bind("<Button-1>", on_click)
        
        row_frame.columnconfigure(1, weight=1)
        
        self.llm_row_count += 1
        
        # Auto-scroll to the bottom
        self.llm_canvas.update_idletasks()
        self.llm_canvas.yview_moveto(1.0)

    def _run_llm_analysis(self, frame, detections, time_str):
        """Run the LLM analysis in a background thread and push results to the UI."""
        img_bytes = None
        try:
            orig_h, orig_w = frame.shape[:2]
            target_w = 320
            scale = target_w / orig_w
            resized_bgr = cv2.resize(frame, (target_w, int(orig_h * scale)))
            is_success, buffer = cv2.imencode(".jpg", resized_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if is_success:
                img_bytes = buffer.tobytes()

            similar_context = []
            frame_rgb = None
            if hasattr(self, 'rag_converter') and self.rag_converter:
                frame_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
                similar_context = self.rag_converter.search_similar(frame_rgb)

            result = self.ai_processor.analyze(frame, detections, similar_context=similar_context)
            # result is None if the call was skipped (cooldown or lock)
            if result is not None:
                self.after(0, self._add_llm_row, time_str, result, img_bytes)
                
                if hasattr(self, 'rag_converter') and self.rag_converter and frame_rgb is not None:
                    try:
                        self.rag_converter.vectorize_and_save(frame_rgb, result, time_str)
                    except Exception as e:
                        print(f"RAG save error: {e}")
        except Exception as e:
            self.after(0, self._add_llm_row, time_str, f"[Error: {e}]", None)

    def _stream_task(self):
        """Continuously pulls frames from the camera generator in the background."""
        try:
            for frame in self.camera_client.get_stream():
                if self.cancel_flag:
                    break
                self.latest_frame = frame
                
                if self.motion_detector and (time.time() - self._start_time >= self._warmup_seconds):
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
            img.thumbnail((480, 360))
            
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
            now = time.time()
            
            if now - self.last_motion_seen_time > 10.0:
                self.last_llm_labels = None
            self.last_motion_seen_time = now
            
            # Only retain targets of interest for RAG and LLM
            ALLOWED_LABELS = {
                'person', 'truck', 'trunk', 'bicycle', 'car', 'animal',
                'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe'
            }
            filtered = [d for d in (self.latest_detections or []) if d['label'] in ALLOWED_LABELS]
            
            # If everything was filtered out, skip this snapshot entirely
            if not filtered:
                self.latest_snapshot = None
                self.latest_original_frame = None
                self.latest_detections = None
                
                # Check for updates again later
                self.after(30, self._update_ui_frame)
                return
            
            # Get 320px JPEG from original frame for saving in DB
            img_bytes = None
            if self.latest_original_frame is not None:
                orig_h, orig_w = self.latest_original_frame.shape[:2]
                target_w = 320
                scale = target_w / orig_w
                resized_bgr = cv2.resize(self.latest_original_frame, (target_w, int(orig_h * scale)))
                is_success, buffer = cv2.imencode(".jpg", resized_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if is_success:
                    img_bytes = buffer.tobytes()

            # Build label text from detections
            label_parts = [f"{d['label']} ({d['confidence']:.0%})" for d in filtered]
            label_text = "\n".join(label_parts[:3])
            
            # Add each detection as a separate row in the log
            for d in filtered:
                self.after(0, self._add_log_row, now_str, d['label'], f"{d['confidence']:.1%}", img_bytes)
            
            self.saved_entries.insert(0, {'photo': new_photo, 'label_text': label_text})
            
            if len(self.saved_entries) > 5:
                self.saved_entries.pop()
                
            for i, entry in enumerate(self.saved_entries):
                self.snapshot_slots[i]['image_label'].config(image=entry['photo'], width=200, height=150)
                self.snapshot_slots[i]['image_label'].image = entry['photo']
                self.snapshot_slots[i]['text_label'].config(text=entry['label_text'])
            
            # Add element to the LLM processing queue
            if self.ai_processor and self.latest_original_frame is not None and filtered:
                current_labels = sorted([d['label'] for d in filtered])
                if current_labels != self.last_llm_labels:
                    self.last_llm_labels = current_labels
                    
                    original_frame = self.latest_original_frame
                    detections_for_llm = filtered
                    
                    # Resize image heavily before queuing to save memory while holding everything
                    if original_frame is not None:
                        orig_h, orig_w = original_frame.shape[:2]
                        target_w = 320
                        scale = target_w / orig_w
                        frame_for_llm = cv2.resize(original_frame, (target_w, int(orig_h * scale)))
                        
                        self.llm_queue.put((frame_for_llm, detections_for_llm, now_str))
                
            self.latest_snapshot = None
            self.latest_original_frame = None
            self.latest_detections = None
            
        self.after(30, self._update_ui_frame)

    def _load_history(self):
        """Load event logs and AI analyses from today and yesterday."""
        if not self.db_client:
            return
        try:
            event_logs = self.db_client.load_event_logs()
            for row in event_logs:
                self._add_log_row(row["time_str"], row["classification"], row["confidence"], image_bytes=None, log_id=row["id"], persist=False)

            ai_analyses = self.db_client.load_ai_analyses()
            for row in ai_analyses:
                self._add_llm_row(row["time_str"], row["result_text"], image_bytes=None, log_id=row["id"], persist=False)
        except Exception as e:
            print(f"[DB] Error loading history: {e}")

    def _add_chat_bubble(self, text, role="user"):
        """Append a chat bubble to the scrollable chat history."""
        is_user = role == "user"
        bg = "#2a4a8a" if is_user else "#1e3a2e"
        fg = "#c8dfff" if is_user else "#a8e6c0"
        prefix = "You: " if is_user else "AI: "
        anchor = "e" if is_user else "w"

        bubble = tk.Label(
            self.chat_scrollable,
            text=f"{prefix}{text}",
            font=("Arial", 10),
            bg=bg, fg=fg,
            wraplength=700,
            justify=tk.LEFT if not is_user else tk.RIGHT,
            anchor=anchor,
            padx=8, pady=4,
            relief="flat",
        )
        bubble.pack(fill=tk.X, pady=2, padx=4)

        self.chat_canvas.update_idletasks()
        self.chat_canvas.yview_moveto(1.0)

    def _send_chat_prompt(self):
        """Read user prompt, enrich with RAG context, send to LLM, display result."""
        prompt = self.chat_entry.get().strip()
        if not prompt:
            return

        self.chat_entry.delete(0, tk.END)
        self._add_chat_bubble(prompt, role="user")
        self._add_chat_bubble("Thinking…", role="ai")

        def run_in_bg():
            import ollama

            rag = getattr(self, 'rag_converter', None)
            context_lines = []
            total_event_count = 0

            if rag:
                try:
                    # Detect if the user is asking a counting/summary question
                    counting_keywords = {'how many', 'count', 'times', 'often', 'frequency',
                                         'total', 'number of', 'how often', 'how much'}
                    is_counting_query = any(kw in prompt.lower() for kw in counting_keywords)

                    # Detect if the user is asking about today/yesterday for filtering
                    date_filter = None
                    if 'today' in prompt.lower():
                        date_filter = datetime.now().date().isoformat()
                    elif 'yesterday' in prompt.lower():
                        from datetime import timedelta
                        date_filter = (datetime.now() - timedelta(days=1)).date().isoformat()

                    # Fetch all historical records (up to a large limit) to ensure we see both today and yesterday
                    all_results = rag.collection.get(include=["metadatas"], limit=10000)
                    all_metas = all_results.get("metadatas", [])
                    # Reverse so we see the most recent events FIRST (prioritize today over yesterday)
                    all_metas_reversed = list(reversed(all_metas))
                    
                    # Apply date filter manually for counting if requested
                    if date_filter:
                        all_metas_reversed = [m for m in all_metas_reversed if m.get('date') == date_filter]

                    total_event_count = sum(1 for m in all_metas_reversed if m and "description" in m)

                    if is_counting_query:
                        # For counting queries: provide a compact summary
                        # Deduplicate similar descriptions to save tokens, but keep full count
                        seen = set()
                        unique_descs = []
                        for m in all_metas_reversed:
                            if m and "description" in m:
                                d = m["description"]
                                # Use first 60 chars as dedup key
                                key = d[:60]
                                if key not in seen:
                                    seen.add(key)
                                    # Include date and time if available
                                    date_tag = m.get('date', '')
                                    comp_ts = f"{date_tag} {m.get('time_str', '?')}".strip()
                                    unique_descs.append(f"[{comp_ts}] {d}")
                                    if len(unique_descs) >= 30:  # increased sample size
                                        break
                        context_lines = unique_descs
                    else:
                        # Use actual semantic vector search for descriptive queries!
                        context_lines = rag.search_by_text(prompt, top_k=12, date_filter=date_filter)

                except Exception as e:
                    print(f"[Chat RAG] error: {e}")

            messages = []
            if context_lines:
                ctx_block = "\n".join(context_lines)
                if is_counting_query if rag else False:
                    system_msg = (
                        f"You are a security camera AI assistant. "
                        f"There are exactly {total_event_count} recorded events in total. "
                        f"Here is a deduplicated sample of the events (duplicates removed for brevity, "
                        f"but the real count is {total_event_count}):\n{ctx_block}\n"
                        f"Answer the user's question using the exact count of {total_event_count}."
                    )
                else:
                    system_msg = (
                        "You are a security camera AI assistant. "
                        f"Total recorded events: {total_event_count}. "
                        "Here are the most relevant past events for this query:\n"
                        f"{ctx_block}\n"
                        "Use this context to answer the user's question."
                    )
                messages.append({'role': 'system', 'content': system_msg})

            messages.append({'role': 'user', 'content': prompt})

            try:
                res = ollama.chat(
                    model='qwen3.5:4b',
                    messages=messages,
                    options={'num_predict': 400, 'temperature': 0.5},
                    think=False,
                    keep_alive='10m',
                )
                if hasattr(res, 'message'):
                    answer = getattr(res.message, 'content', '') or ''
                else:
                    answer = res.get('message', {}).get('content', '') or ''
                answer = answer.strip() or "(no response)"
            except Exception as e:
                answer = f"[Error: {e}]"

            # Replace "Thinking…" bubble with the real answer on the main thread
            def update_ui():
                # Remove the last bubble ("Thinking…") and add the real answer
                children = self.chat_scrollable.winfo_children()
                if children:
                    children[-1].destroy()
                self._add_chat_bubble(answer, role="ai")

            self.after(0, update_ui)

        threading.Thread(target=run_in_bg, daemon=True).start()

    def close_app(self):
        self.cancel_flag = True
        if self.db_client:
            self.db_client.close()
        self.destroy()
