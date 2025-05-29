import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor
import cv2

# 初始化SAM模型（确保你安装了 segment_anything 库）
device = torch.device("cpu")
# 加载SAM模型
sam_model = sam_model_registry["vit_b"](checkpoint="sam_vit_b.pth").to(device)
predictor = SamPredictor(sam_model)
# 合成图片保存位置
save_folder = r"E:\FreeControl\5.21"

# 使用说明
# 1.
# 左键点击，点选
# 右键拖动，框选
# 2.
# 在背景图上，鼠标左键拖动f
# ctrl+鼠标滚轮，可以调整大小


class ImageEditor:
    def __init__(self, root):
        self.root = root
        self.root.title("SAM 前景抠图与背景合成工具")

        # 初始化变量
        self.input_pil = None
        self.input_np = None
        self.bg_pil = None
        self.bg_tk = None
        self.fg_pil = None
        self.fg_preview_pil = None
        self.fg_tk = None
        self.fg_preview_tk = None
        self.drag_data = {"x": 0, "y": 0, "item": None, "scale": 1.0}

        # 左：原图区
        self.input_canvas = tk.Canvas(root, width=320, height=320, bg="gray")
        self.input_canvas.grid(row=0, column=0)
        self.input_canvas.bind("<Button-1>", self.on_click)
        self.input_canvas.bind("<ButtonPress-3>", self.on_drag_start_rect)
        self.input_canvas.bind("<B3-Motion>", self.on_drag_move_rect)
        self.input_canvas.bind("<ButtonRelease-3>", self.on_drag_release_rect)

        # 中：背景图 + 合成预览
        self.bg_canvas = tk.Canvas(root, width=512, height=512, bg="black")
        self.bg_canvas.grid(row=0, column=1)
        self.bg_canvas.bind("<ButtonPress-1>", self.on_fg_drag_start)
        self.bg_canvas.bind("<B1-Motion>", self.on_fg_drag_move)
        self.bg_canvas.bind("<Control-MouseWheel>", self.on_zoom)

        # 右：前景图预览
        self.fg_canvas = tk.Canvas(root, width=320, height=320, bg="gray")
        self.fg_canvas.grid(row=0, column=2)

        # 按钮区
        btn_frame = tk.Frame(root)
        btn_frame.grid(row=1, column=0, columnspan=3)
        tk.Button(btn_frame, text="加载原图", command=self.load_input).pack(side=tk.LEFT)
        tk.Button(btn_frame, text="加载背景图", command=self.load_background).pack(side=tk.LEFT)
        tk.Button(btn_frame, text="保存合成图", command=self.save_image).pack(side=tk.LEFT)

        self.rect_start = None
        self.rect_id = None

    def load_input(self):
        path = filedialog.askopenfilename()
        if not path: return
        self.input_pil = Image.open(path).convert("RGB").resize((256,256))
        self.input_np = np.array(Image.open(path).convert("RGB").resize((1024,1024)))
        self.input_imgtk = ImageTk.PhotoImage(self.input_pil.resize((320, 320)))
        self.input_canvas.create_image(0, 0, anchor="nw", image=self.input_imgtk)

    def load_background(self):
        path = filedialog.askopenfilename()
        if not path: return
        self.bg_pil = Image.open(path).convert("RGBA").resize((1024,1024))
        # self.bg_tk = ImageTk.PhotoImage(self.bg_pil.resize(512, 512))
        self.bg_tk = ImageTk.PhotoImage(self.bg_pil.resize((512, 512)))
        self.bg_canvas.delete("all")
        self.bg_canvas.create_image(0, 0, anchor="nw", image=self.bg_tk)
        self.drag_data["item"] = None

    def on_click(self, event):
        if self.input_np is None: return
        h, w = self.input_np.shape[:2]
        scale_x = w / 320
        scale_y = h / 320
        x = int(event.x * scale_x)
        y = int(event.y * scale_y)
        self.run_sam_click(x, y)

    def on_drag_start_rect(self, event):
        self.rect_start = (event.x, event.y)
        if self.rect_id:
            self.input_canvas.delete(self.rect_id)
        self.rect_id = self.input_canvas.create_rectangle(event.x, event.y, event.x, event.y, outline="red")

    def on_drag_move_rect(self, event):
        if not self.rect_start: return
        self.input_canvas.coords(self.rect_id, self.rect_start[0], self.rect_start[1], event.x, event.y)

    def on_drag_release_rect(self, event):
        if not self.rect_start: return
        x0, y0 = self.rect_start
        x1, y1 = event.x, event.y
        self.rect_start = None
        self.input_canvas.delete(self.rect_id)
        h, w = self.input_np.shape[:2]
        scale_x = w / 320
        scale_y = h / 320
        box = [int(x0 * scale_x), int(y0 * scale_y), int(x1 * scale_x), int(y1 * scale_y)]
        self.run_sam_box(box)

    def run_sam_click(self, x, y):
        # 使用SAM模型进行点击抠图
        print(self.input_np.shape)
        predictor.set_image(self.input_np)
        input_point = np.array([[x, y]])
        input_label = np.array([1])
        masks, _, _ = predictor.predict(point_coords=input_point, point_labels=input_label, multimask_output=False)
        mask = masks[0]
        self.extract_foreground(mask)

    def run_sam_box(self, box):
        # 使用SAM模型进行框选抠图
        print(self.input_np.shape)
        print(box)
        predictor.set_image(self.input_np)
        masks, _, _ = predictor.predict(box=np.array([box]), multimask_output=False)
        mask = masks[0]
        self.extract_foreground(mask)

    def extract_foreground(self, mask):
        # 提取前景
        mask = cv2.resize(mask.astype(np.uint8), (self.input_np.shape[1], self.input_np.shape[0]))
        fg = self.input_np * mask[..., None]
        # 为了确保透明区域透明，使用mask生成透明度通道
        alpha_channel = np.zeros_like(mask, dtype=np.uint8)
        alpha_channel[mask == 1] = 255  # 前景区域完全不透明，背景区域透明
        # 将前景图像与透明度通道合并
        fg_rgba = np.dstack((fg, alpha_channel))  # 将前三个通道（RGB）和透明度（A）合并成RGBA格式
        # 转换为PIL图像
        self.fg_pil = Image.fromarray(fg_rgba).convert("RGBA")
        print("前景提取完成")
        self.update_fg_display()

    def update_fg_display(self):
        self.fg_tk = ImageTk.PhotoImage(self.fg_pil.resize((320, 320)))
        self.fg_canvas.delete("all")
        self.fg_canvas.create_image(0, 0, anchor="nw", image=self.fg_tk)

        if self.bg_tk:
            # self.bg_canvas.delete("fg")
            self.fg_preview_pil = self.fg_pil.resize((200, 200))
            self.fg_preview_tk = ImageTk.PhotoImage(self.fg_preview_pil)
            self.drag_data["item"] = self.bg_canvas.create_image(50, 50, anchor="nw", image=self.fg_preview_tk, tags="fg")
            self.drag_data["scale"] = 1.0

    def on_fg_drag_start(self, event):
        self.drag_data["x"] = event.x
        self.drag_data["y"] = event.y

    def on_fg_drag_move(self, event):
        dx = event.x - self.drag_data["x"]
        dy = event.y - self.drag_data["y"]
        self.drag_data["x"] = event.x
        self.drag_data["y"] = event.y
        if self.drag_data["item"]:
            self.bg_canvas.move(self.drag_data["item"], dx, dy)

    def on_zoom(self, event):
        if not self.fg_pil or not self.drag_data["item"]:
            return
        if event.delta > 0:
            self.drag_data["scale"] *= 1.1
        else:
            self.drag_data["scale"] *= 0.9
        new_size = int(200 * self.drag_data["scale"])

        self.fg_preview_pil = self.fg_pil.resize((new_size, new_size), Image.Resampling.LANCZOS)  # ✅ 更新 preview_pil
        self.fg_preview_tk = ImageTk.PhotoImage(self.fg_preview_pil)
        self.bg_canvas.itemconfig(self.drag_data["item"], image=self.fg_preview_tk)

    def save_image(self):
        if self.bg_pil and self.fg_pil and self.drag_data["item"]:
            bg_copy = self.bg_pil.copy().convert("RGBA")
            fg_orig = self.fg_pil.copy().convert("RGBA")

            # 画布显示尺寸和实际背景图尺寸比例
            canvas_display_size = 512
            bg_actual_width, bg_actual_height = self.bg_pil.size
            scale_factor_x = bg_actual_width / canvas_display_size
            scale_factor_y = bg_actual_height / canvas_display_size

            # 获取画布上前景图的位置
            canvas_coords = self.bg_canvas.coords(self.drag_data["item"])
            if not canvas_coords:
                return
            x_canvas, y_canvas = canvas_coords
            x_bg = int(x_canvas * scale_factor_x)
            y_bg = int(y_canvas * scale_factor_y)

            # 获取当前前景图的实际大小（根据用户缩放倍数）
            scale = self.drag_data["scale"]
            fg_width = int(200 * scale * scale_factor_x)
            fg_height = int(200 * scale * scale_factor_y)
            fg_resized = fg_orig.resize((fg_width, fg_height), Image.Resampling.LANCZOS)

            # 粘贴合成图
            bg_copy.paste(fg_resized, (x_bg, y_bg), fg_resized)

            # 保存
            os.makedirs(save_folder, exist_ok=True)
            n = len(os.listdir(save_folder))
            save_path = os.path.join(save_folder, f"{ n + 1}.png")
            print(f"保存路径: {save_path}")
            bg_copy.save(save_path)
            messagebox.showinfo("保存成功", f"合成图已保存为 {save_path}")


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageEditor(root)
    root.mainloop()
