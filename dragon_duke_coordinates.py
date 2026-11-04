"""Plan a straight-line Dragon Duke route from a Clash of Clans screenshot."""

from __future__ import annotations

import argparse
import base64
import math
import random
import sys
from pathlib import Path
from typing import Sequence
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageGrab


Point = tuple[float, float]
ImageBounds = tuple[int, int, int, int]
EPSILON = 1e-9

NEON_MAGENTA = (255, 35, 235)  # OpenCV uses BGR channel order.
NEON_CYAN = (255, 255, 0)
NEON_YELLOW = (0, 255, 255)
OUTLINE = (20, 18, 25)
WHITE = (245, 245, 245)
CLIPBOARD_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}

APP_BG = "#121719"
PANEL_BG = "#1a2224"
CARD_BG = "#222c2e"
BORDER = "#354244"
TEXT = "#edf0e8"
MUTED = "#a5b0a9"
GOLD = "#d7b76b"
CORAL = "#bd624d"
CYAN = "#58cbc7"
YELLOW = "#e8c873"


def _validated_point(
    name: str,
    point: Sequence[float],
    right: float,
    bottom: float,
) -> Point:
    """Return a finite point inside the image's pixel-coordinate bounds."""
    try:
        x, y = point
        x = float(x)
        y = float(y)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain two numeric coordinates.") from exc

    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError(f"{name} coordinates must be finite.")
    if x < -EPSILON or x > right + EPSILON or y < -EPSILON or y > bottom + EPSILON:
        raise ValueError(f"{name} must lie inside the image.")

    return min(max(x, 0.0), right), min(max(y, 0.0), bottom)


def line_edge_intersections(
    width: int,
    height: int,
    center: Point,
    target: Point,
) -> tuple[Point, Point]:
    """Find both image-edge pixels on the infinite line through center and target.

    Pixel coordinates span 0..width-1 and 0..height-1. The GUI's automatic
    center follows the requested image_width/2, image_height/2 convention.
    """
    if width < 2 or height < 2:
        raise ValueError("Image must be at least 2 pixels wide and 2 pixels high.")

    right = float(width - 1)
    bottom = float(height - 1)
    cx, cy = _validated_point("Center", center, right, bottom)
    tx, ty = _validated_point("Target", target, right, bottom)
    dx = tx - cx
    dy = ty - cy

    if math.hypot(dx, dy) <= EPSILON:
        raise ValueError("Target coincides with center; click a point away from center.")

    # Parametrize the line as P(t) = C + t(T - C). Intersect each rectangle
    # edge, skip zero direction components, and keep only points on the image.
    candidates: list[tuple[float, Point]] = []

    def add_candidate(t: float, x: float, y: float) -> None:
        if -EPSILON <= x <= right + EPSILON and -EPSILON <= y <= bottom + EPSILON:
            point = (min(max(x, 0.0), right), min(max(y, 0.0), bottom))
            if all(math.dist(point, existing) > EPSILON for _, existing in candidates):
                candidates.append((t, point))

    if abs(dx) > EPSILON:
        for x in (0.0, right):
            t = (x - cx) / dx
            add_candidate(t, x, cy + t * dy)

    if abs(dy) > EPSILON:
        for y in (0.0, bottom):
            t = (y - cy) / dy
            add_candidate(t, cx + t * dx, y)

    if len(candidates) != 2:
        raise ValueError("Could not find two distinct image-edge intersections.")

    candidates.sort(key=lambda item: item[0])
    return candidates[0][1], candidates[1][1]


def canvas_to_image_point(
    canvas_x: int,
    canvas_y: int,
    bounds: ImageBounds,
    image_width: int,
    image_height: int,
) -> Point | None:
    """Map a canvas click to the source image, or return None outside the image."""
    left, top, display_width, display_height = bounds
    if display_width < 1 or display_height < 1:
        return None
    if not (left <= canvas_x < left + display_width):
        return None
    if not (top <= canvas_y < top + display_height):
        return None

    x_scale = (image_width - 1) / max(display_width - 1, 1)
    y_scale = (image_height - 1) / max(display_height - 1, 1)
    x = (canvas_x - left) * x_scale
    y = (canvas_y - top) * y_scale
    return min(max(x, 0.0), image_width - 1.0), min(max(y, 0.0), image_height - 1.0)


def _raster_point(point: Point, width: int, height: int) -> tuple[int, int]:
    """Round and clamp a geometric point to a drawable image pixel."""
    x = min(max(int(round(point[0])), 0), width - 1)
    y = min(max(int(round(point[1])), 0), height - 1)
    return x, y


def _draw_dashed_line(image: np.ndarray, start: Point, end: Point) -> None:
    """Draw a neon dashed segment with a dark outline for contrast."""
    start_array = np.asarray(start, dtype=np.float64)
    end_array = np.asarray(end, dtype=np.float64)
    direction = end_array - start_array
    length = float(np.linalg.norm(direction))
    if length <= EPSILON:
        return

    dash_length = max(10.0, min(30.0, length / 24.0))
    gap_length = max(6.0, dash_length * 0.55)
    offset = 0.0
    height, width = image.shape[:2]

    while offset < length:
        end_offset = min(offset + dash_length, length)
        first = start_array + direction * (offset / length)
        second = start_array + direction * (end_offset / length)
        p1 = _raster_point((float(first[0]), float(first[1])), width, height)
        p2 = _raster_point((float(second[0]), float(second[1])), width, height)
        cv2.line(image, p1, p2, OUTLINE, 7, cv2.LINE_AA)
        cv2.line(image, p1, p2, NEON_MAGENTA, 4, cv2.LINE_AA)
        cv2.line(image, p1, p2, WHITE, 1, cv2.LINE_AA)
        offset += dash_length + gap_length


def draw_trajectory_overlay(
    image: np.ndarray,
    center: Point,
    target: Point,
    endpoints: tuple[Point, Point],
) -> np.ndarray:
    """Draw the projected route, center, target, and both deployment zones."""
    overlay = image.copy()
    height, width = overlay.shape[:2]

    _draw_dashed_line(overlay, endpoints[0], endpoints[1])

    center_pixel = _raster_point(center, width, height)
    cv2.drawMarker(
        overlay,
        center_pixel,
        OUTLINE,
        cv2.MARKER_TILTED_CROSS,
        19,
        5,
        cv2.LINE_AA,
    )
    cv2.drawMarker(
        overlay,
        center_pixel,
        WHITE,
        cv2.MARKER_TILTED_CROSS,
        17,
        2,
        cv2.LINE_AA,
    )

    target_pixel = _raster_point(target, width, height)
    cv2.circle(overlay, target_pixel, 9, OUTLINE, 5, cv2.LINE_AA)
    cv2.circle(overlay, target_pixel, 9, NEON_YELLOW, 2, cv2.LINE_AA)

    zone_colors = (NEON_CYAN, NEON_YELLOW)
    for endpoint, color in zip(endpoints, zone_colors):
        pixel = _raster_point(endpoint, width, height)
        cv2.drawMarker(overlay, pixel, OUTLINE, cv2.MARKER_CROSS, 31, 7, cv2.LINE_AA)
        cv2.drawMarker(overlay, pixel, color, cv2.MARKER_CROSS, 27, 3, cv2.LINE_AA)
        cv2.circle(overlay, pixel, 8, OUTLINE, 5, cv2.LINE_AA)
        cv2.circle(overlay, pixel, 8, color, 2, cv2.LINE_AA)

    return overlay


def load_image(path: str | Path) -> np.ndarray:
    """Read a supported image, including paths with non-ASCII characters."""
    image_path = Path(path)
    encoded = np.fromfile(image_path, dtype=np.uint8)
    if encoded.size == 0:
        raise ValueError(f"File is empty: {image_path}")

    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ValueError(f"OpenCV could not decode this image: {image_path}")
    if image.shape[1] < 2 or image.shape[0] < 2:
        raise ValueError("Image must be at least 2 pixels wide and 2 pixels high.")
    return image


def load_clipboard_image() -> tuple[np.ndarray, str, Path | None]:
    """Read an image or copied image file from the Windows clipboard."""
    clipboard_data = ImageGrab.grabclipboard()

    if isinstance(clipboard_data, Image.Image):
        rgb = np.asarray(clipboard_data.convert("RGB"))
        image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if image.shape[1] < 2 or image.shape[0] < 2:
            raise ValueError("Clipboard image must be at least 2 pixels wide and high.")
        return image, "Clipboard image", None

    if isinstance(clipboard_data, list):
        for item in clipboard_data:
            path = Path(str(item))
            if path.suffix.lower() in CLIPBOARD_IMAGE_EXTENSIONS:
                return load_image(path), path.name, path

    raise ValueError("Clipboard has no image. Copy an image, then press Ctrl+V.")


class DragonDukePlanner:
    """Tkinter interface for selecting a target and projecting its center axis."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Where Will the Duke Go?")
        self.root.geometry("1380x860")
        self.root.minsize(1040, 680)
        self._configure_style()
        self.root.configure(background=APP_BG)

        self.image: np.ndarray | None = None
        self.image_path: Path | None = None
        self.center: Point | None = None
        self.target: Point | None = None
        self.endpoints: tuple[Point, Point] | None = None
        self.center_is_manual = False
        self.calibrating_center = False
        self.display_bounds: ImageBounds | None = None
        self.photo: tk.PhotoImage | None = None
        self.render_job: str | None = None

        self.center_var = tk.StringVar(value="—")
        self.target_var = tk.StringVar(value="—")
        self.zone_a_var = tk.StringVar(value="—")
        self.zone_b_var = tk.StringVar(value="—")
        self.image_var = tk.StringVar(value="No image loaded")
        self.status_var = tk.StringVar(value="Open a base screenshot to begin.")

        self._build_layout()
        self._update_controls()
        self.canvas.bind("<Button-1>", self._on_canvas_click)
        self.canvas.bind("<Configure>", self._schedule_render)
        self.root.bind_all("<Control-v>", self.paste_clipboard_image)
        self._render_canvas()

    def _configure_style(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure("App.TFrame", background=APP_BG)
        style.configure("Panel.TFrame", background=PANEL_BG)
        style.configure(
            "Card.TFrame",
            background=CARD_BG,
            bordercolor=BORDER,
            relief="solid",
            borderwidth=1,
        )
        style.configure("TSeparator", background=BORDER)
        style.configure(
            "Eyebrow.TLabel",
            background=APP_BG,
            foreground=GOLD,
            font=("Segoe UI", 9, "bold"),
        )
        style.configure(
            "Title.TLabel",
            background=APP_BG,
            foreground=TEXT,
            font=("Segoe UI", 25, "bold"),
        )
        style.configure(
            "Subtitle.TLabel",
            background=APP_BG,
            foreground=MUTED,
            font=("Segoe UI", 10),
        )
        style.configure(
            "Section.TLabel",
            background=PANEL_BG,
            foreground=GOLD,
            font=("Segoe UI", 9, "bold"),
        )
        style.configure(
            "PanelBody.TLabel",
            background=PANEL_BG,
            foreground=TEXT,
            font=("Segoe UI", 10),
        )
        style.configure(
            "Muted.TLabel",
            background=PANEL_BG,
            foreground=MUTED,
            font=("Segoe UI", 9),
        )
        style.configure(
            "Coordinate.TLabel",
            background=CARD_BG,
            foreground=TEXT,
            font=("Consolas", 10),
        )
        style.configure(
            "ZoneA.TLabel",
            background=CARD_BG,
            foreground=CYAN,
            font=("Segoe UI", 9, "bold"),
        )
        style.configure(
            "ZoneB.TLabel",
            background=CARD_BG,
            foreground=YELLOW,
            font=("Segoe UI", 9, "bold"),
        )
        style.configure(
            "Status.TLabel",
            background=CARD_BG,
            foreground=MUTED,
            font=("Segoe UI", 9),
        )
        style.configure(
            "Primary.TButton",
            background=CORAL,
            foreground="#fff8f1",
            padding=(12, 10),
            borderwidth=0,
            focusthickness=2,
            focuscolor=GOLD,
            font=("Segoe UI", 10, "bold"),
        )
        style.map(
            "Primary.TButton",
            background=[("disabled", "#384141"), ("pressed", "#994936"), ("active", "#d0735c")],
            foreground=[("disabled", "#79847e")],
        )
        style.configure(
            "Secondary.TButton",
            background=CARD_BG,
            foreground=TEXT,
            padding=(12, 9),
            borderwidth=1,
            relief="solid",
            focusthickness=2,
            focuscolor=GOLD,
            font=("Segoe UI", 10, "bold"),
        )
        style.map(
            "Secondary.TButton",
            background=[("disabled", "#22292a"), ("pressed", "#293334"), ("active", "#303b3d")],
            foreground=[("disabled", "#74807b")],
        )
        style.configure(
            "Quiet.TButton",
            background=PANEL_BG,
            foreground=MUTED,
            padding=(10, 8),
            borderwidth=1,
            relief="solid",
            focusthickness=2,
            focuscolor=GOLD,
            font=("Segoe UI", 9, "bold"),
        )
        style.map(
            "Quiet.TButton",
            background=[("disabled", PANEL_BG), ("pressed", "#202a2c"), ("active", "#263133")],
            foreground=[("disabled", "#69736e"), ("active", TEXT)],
        )

    def _build_layout(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        shell = ttk.Frame(self.root, style="App.TFrame", padding=(20, 16))
        shell.grid(row=0, column=0, sticky="nsew")
        shell.columnconfigure(0, weight=1)
        shell.rowconfigure(1, weight=1)

        header = ttk.Frame(shell, style="App.TFrame")
        header.grid(row=0, column=0, sticky="ew", pady=(0, 16))
        header.columnconfigure(0, weight=1)
        ttk.Label(
            header,
            text="DRAGON DUKE  /  ROCKET BACKPACK",
            style="Eyebrow.TLabel",
        ).grid(row=0, column=0, sticky="w", pady=(0, 4))
        ttk.Label(
            header,
            text="Where Will the Duke Go?",
            style="Title.TLabel",
        ).grid(row=1, column=0, sticky="w")
        ttk.Label(
            header,
            text="Select a building to project the straight route through the village center.",
            style="Subtitle.TLabel",
        ).grid(row=2, column=0, sticky="w", pady=(3, 0))

        workspace = ttk.Frame(shell, style="App.TFrame")
        workspace.grid(row=1, column=0, sticky="nsew")
        workspace.columnconfigure(0, weight=1)
        workspace.columnconfigure(1, minsize=344)
        workspace.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(
            workspace,
            background="#0d1214",
            highlightthickness=1,
            highlightbackground=BORDER,
            highlightcolor=GOLD,
            width=960,
            height=700,
        )
        self.canvas.grid(row=0, column=0, sticky="nsew", padx=(0, 16))

        panel = ttk.Frame(workspace, style="Panel.TFrame", padding=18, width=328)
        panel.grid(row=0, column=1, sticky="nsew")
        panel.grid_propagate(False)
        panel.columnconfigure(0, weight=1)

        ttk.Label(panel, text="SCREENSHOT", style="Section.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 8)
        )
        ttk.Button(
            panel,
            text="Open image…",
            command=self.open_image,
            style="Secondary.TButton",
        ).grid(row=1, column=0, sticky="ew", pady=3)
        ttk.Button(
            panel,
            text="Paste image  ·  Ctrl+V",
            command=self.paste_clipboard_image,
            style="Primary.TButton",
        ).grid(row=2, column=0, sticky="ew", pady=3)
        ttk.Label(
            panel,
            textvariable=self.image_var,
            style="Muted.TLabel",
            wraplength=288,
            justify="left",
        ).grid(row=3, column=0, sticky="ew", pady=(6, 10))

        ttk.Separator(panel).grid(row=4, column=0, sticky="ew", pady=5)
        ttk.Label(panel, text="CENTER POINT", style="Section.TLabel").grid(
            row=5, column=0, sticky="w", pady=(9, 3)
        )
        ttk.Label(
            panel,
            textvariable=self.center_var,
            style="PanelBody.TLabel",
            wraplength=288,
        ).grid(row=6, column=0, sticky="w", pady=(0, 5))
        self.center_button = ttk.Button(
            panel,
            text="Set center by click",
            command=self.toggle_center_calibration,
            style="Secondary.TButton",
        )
        self.center_button.grid(row=7, column=0, sticky="ew", pady=3)
        self.image_center_button = ttk.Button(
            panel,
            text="Use image midpoint",
            command=self.use_image_center,
            style="Quiet.TButton",
        )
        self.image_center_button.grid(row=8, column=0, sticky="ew", pady=3)

        ttk.Label(panel, text="TARGET", style="Section.TLabel").grid(
            row=9, column=0, sticky="w", pady=(12, 3)
        )
        ttk.Label(
            panel,
            textvariable=self.target_var,
            style="PanelBody.TLabel",
        ).grid(row=10, column=0, sticky="w", pady=(0, 4))
        self.clear_button = ttk.Button(
            panel,
            text="Clear target",
            command=self.clear_target,
            style="Quiet.TButton",
        )
        self.clear_button.grid(row=11, column=0, sticky="ew", pady=3)

        ttk.Separator(panel).grid(row=12, column=0, sticky="ew", pady=10)
        ttk.Label(panel, text="IMAGE EDGE INTERSECTIONS", style="Section.TLabel").grid(
            row=13, column=0, sticky="w", pady=(0, 6)
        )

        for row, label, variable, label_style in (
            (14, "ZONE A", self.zone_a_var, "ZoneA.TLabel"),
            (15, "ZONE B", self.zone_b_var, "ZoneB.TLabel"),
        ):
            card = ttk.Frame(panel, style="Card.TFrame", padding=(10, 9))
            card.grid(row=row, column=0, sticky="ew", pady=3)
            card.columnconfigure(1, weight=1)
            ttk.Label(card, text=label, style=label_style).grid(
                row=0, column=0, sticky="w", padx=(0, 14)
            )
            ttk.Label(card, textvariable=variable, style="Coordinate.TLabel").grid(
                row=0, column=1, sticky="e"
            )

        panel.rowconfigure(17, weight=1)
        status_card = ttk.Frame(panel, style="Card.TFrame", padding=(10, 9))
        status_card.grid(row=17, column=0, sticky="ew", pady=(12, 0))
        ttk.Label(
            status_card,
            textvariable=self.status_var,
            style="Status.TLabel",
            wraplength=286,
            justify="left",
        ).grid(row=0, column=0, sticky="w")

    def _update_controls(self) -> None:
        has_image = self.image is not None
        state = "normal" if has_image else "disabled"
        self.center_button.configure(state=state)
        self.image_center_button.configure(state=state)
        self.clear_button.configure(state="normal" if self.target is not None else "disabled")
        self.center_button.configure(
            text="Cancel center click" if self.calibrating_center else "Set center by click"
        )

    def open_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose a Clash of Clans screenshot",
            filetypes=(
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff"),
                ("All files", "*.*"),
            ),
        )
        if not path:
            return

        try:
            image = load_image(path)
        except (OSError, ValueError, cv2.error) as exc:
            messagebox.showerror("Could not open image", str(exc), parent=self.root)
            return

        image_path = Path(path)
        self._set_image(image, image_path.name, image_path)

    def paste_clipboard_image(self, _event: tk.Event | None = None) -> str:
        """Load the current clipboard image and handle Ctrl+V globally."""
        try:
            image, display_name, image_path = load_clipboard_image()
        except (OSError, ValueError, cv2.error, NotImplementedError) as exc:
            self.status_var.set(str(exc))
            return "break"

        self._set_image(image, display_name, image_path)
        self.status_var.set("Pasted image. Click a defense to draw its axis.")
        return "break"

    def _set_image(
        self,
        image: np.ndarray,
        display_name: str,
        image_path: Path | None,
    ) -> None:
        """Install a new image and reset target and calibration state."""
        self.image = image
        self.image_path = image_path
        height, width = image.shape[:2]
        self.center = (width / 2.0, height / 2.0)
        self.target = None
        self.endpoints = None
        self.center_is_manual = False
        self.calibrating_center = False
        self.image_var.set(f"{display_name}\n{width} × {height} px")
        self.status_var.set("Image center set automatically. Click a defense to draw its axis.")
        self._update_readouts()
        self._update_controls()
        self._schedule_render()

    def toggle_center_calibration(self) -> None:
        if self.image is None:
            return
        self.calibrating_center = not self.calibrating_center
        if self.calibrating_center:
            self.status_var.set("Click the village's true center in the image.")
        else:
            self.status_var.set("Center calibration canceled.")
        self._update_controls()

    def use_image_center(self) -> None:
        if self.image is None:
            return
        height, width = self.image.shape[:2]
        self.center = (width / 2.0, height / 2.0)
        self.center_is_manual = False
        self.calibrating_center = False
        self.status_var.set("Using the image midpoint as village center.")
        self._recalculate()
        self._update_controls()

    def clear_target(self) -> None:
        self.target = None
        self.endpoints = None
        self.calibrating_center = False
        self.status_var.set("Target cleared. Click another defense to try a new axis.")
        self._update_readouts()
        self._update_controls()
        self._schedule_render()

    def _on_canvas_click(self, event: tk.Event) -> None:
        if self.image is None or self.display_bounds is None:
            return

        height, width = self.image.shape[:2]
        point = canvas_to_image_point(
            event.x,
            event.y,
            self.display_bounds,
            width,
            height,
        )
        if point is None:
            return

        if self.calibrating_center:
            self.center = point
            self.center_is_manual = True
            self.calibrating_center = False
            self.status_var.set("Manual center set. Click a defense to project its route.")
            self._recalculate()
        else:
            self.target = point
            self._recalculate()
        self._update_controls()

    def _recalculate(self) -> None:
        if self.center is None or self.target is None:
            self.endpoints = None
            self._update_readouts()
            self._schedule_render()
            return

        try:
            assert self.image is not None
            height, width = self.image.shape[:2]
            self.endpoints = line_edge_intersections(width, height, self.center, self.target)
        except ValueError as exc:
            self.endpoints = None
            self.status_var.set(str(exc))
        else:
            self.status_var.set("Projected straight line through center and target.")

        self._update_readouts()
        self._schedule_render()

    def _update_readouts(self) -> None:
        if self.center is None:
            self.center_var.set("—")
        else:
            source = "manual" if self.center_is_manual else "image midpoint"
            self.center_var.set(f"({self.center[0]:.2f}, {self.center[1]:.2f}) px · {source}")

        self.target_var.set("—" if self.target is None else self._format_point(self.target))
        if self.endpoints is None:
            self.zone_a_var.set("—")
            self.zone_b_var.set("—")
        else:
            self.zone_a_var.set(self._format_point(self.endpoints[0]))
            self.zone_b_var.set(self._format_point(self.endpoints[1]))

    @staticmethod
    def _format_point(point: Point) -> str:
        return f"({point[0]:.2f}, {point[1]:.2f}) px"

    def _schedule_render(self, _event: tk.Event | None = None) -> None:
        if self.render_job is not None:
            self.root.after_cancel(self.render_job)
        self.render_job = self.root.after(25, self._render_canvas)

    def _render_canvas(self) -> None:
        self.render_job = None
        self.canvas.delete("all")
        canvas_width = max(self.canvas.winfo_width(), 2)
        canvas_height = max(self.canvas.winfo_height(), 2)

        if self.image is None:
            self.display_bounds = None
            self.canvas.create_text(
                canvas_width / 2,
                canvas_height / 2 - 13,
                text="Open a base image to start",
                fill=TEXT,
                font=("Segoe UI", 16, "bold"),
                justify="center",
            )
            self.canvas.create_text(
                canvas_width / 2,
                canvas_height / 2 + 17,
                text="or paste one with Ctrl+V",
                fill=MUTED,
                font=("Segoe UI", 10),
                justify="center",
            )
            return

        source_height, source_width = self.image.shape[:2]
        available_width = max(canvas_width - 32, 2)
        available_height = max(canvas_height - 32, 2)
        scale = min(
            1.0,
            available_width / source_width,
            available_height / source_height,
        )
        display_width = max(2, min(available_width, int(round(source_width * scale))))
        display_height = max(2, min(available_height, int(round(source_height * scale))))
        left = (canvas_width - display_width) // 2
        top = (canvas_height - display_height) // 2
        self.display_bounds = (left, top, display_width, display_height)

        if (display_width, display_height) == (source_width, source_height):
            preview = self.image.copy()
        else:
            preview = cv2.resize(
                self.image,
                (display_width, display_height),
                interpolation=cv2.INTER_AREA,
            )

        x_scale = (display_width - 1) / (source_width - 1)
        y_scale = (display_height - 1) / (source_height - 1)

        def to_preview(point: Point) -> Point:
            return point[0] * x_scale, point[1] * y_scale

        if self.center is not None:
            preview_center = to_preview(self.center)
            center_pixel = _raster_point(preview_center, display_width, display_height)
            cv2.drawMarker(
                preview,
                center_pixel,
                WHITE,
                cv2.MARKER_TILTED_CROSS,
                15,
                2,
                cv2.LINE_AA,
            )

        if self.target is not None and self.endpoints is not None and self.center is not None:
            preview = draw_trajectory_overlay(
                preview,
                to_preview(self.center),
                to_preview(self.target),
                (to_preview(self.endpoints[0]), to_preview(self.endpoints[1])),
            )
        elif self.target is not None:
            target_pixel = _raster_point(to_preview(self.target), display_width, display_height)
            cv2.circle(preview, target_pixel, 8, NEON_YELLOW, 2, cv2.LINE_AA)

        encoded_ok, encoded = cv2.imencode(".png", preview)
        if not encoded_ok:
            self.status_var.set("Could not render the image preview.")
            return

        self.photo = tk.PhotoImage(data=base64.b64encode(encoded.tobytes()).decode("ascii"))
        self.canvas.create_image(left, top, anchor=tk.NW, image=self.photo)
        self.canvas.create_rectangle(
            left,
            top,
            left + display_width - 1,
            top + display_height - 1,
            outline=BORDER,
            width=1,
        )


def _assert_intersections(
    width: int,
    height: int,
    center: Point,
    target: Point,
    intersections: tuple[Point, Point],
) -> None:
    """Assert that the returned pair is bounded, distinct, and collinear."""
    right = width - 1.0
    bottom = height - 1.0
    if len(intersections) != 2 or math.dist(*intersections) <= EPSILON:
        raise AssertionError("Expected two distinct intersections.")

    dx = target[0] - center[0]
    dy = target[1] - center[1]
    scale = max(1.0, math.hypot(dx, dy))
    for x, y in intersections:
        if not (-EPSILON <= x <= right + EPSILON and -EPSILON <= y <= bottom + EPSILON):
            raise AssertionError(f"Intersection is out of bounds: {(x, y)}")
        on_edge = (
            abs(x) <= EPSILON
            or abs(x - right) <= EPSILON
            or abs(y) <= EPSILON
            or abs(y - bottom) <= EPSILON
        )
        if not on_edge:
            raise AssertionError(f"Intersection is not on an image edge: {(x, y)}")
        cross_product = dx * (y - center[1]) - dy * (x - center[0])
        if abs(cross_product) > 1e-6 * scale:
            raise AssertionError(f"Intersection is not on the requested axis: {(x, y)}")


def run_self_test(image_path: str | Path) -> int:
    """Run geometry and OpenCV drawing checks without creating a Tk window."""
    try:
        image = load_image(image_path)
        height, width = image.shape[:2]
        center = (width / 2.0, height / 2.0)
        rng = random.Random(20261104)

        while True:
            target = (
                rng.uniform(0.15 * (width - 1), 0.85 * (width - 1)),
                rng.uniform(0.15 * (height - 1), 0.85 * (height - 1)),
            )
            if math.dist(center, target) > max(width, height) * 0.05:
                break

        random_endpoints = line_edge_intersections(width, height, center, target)
        _assert_intersections(width, height, center, target, random_endpoints)
        rendered = draw_trajectory_overlay(image, center, target, random_endpoints)
        if rendered.shape != image.shape or np.array_equal(rendered, image):
            raise AssertionError("OpenCV overlay did not render on the test image.")

        right = width - 1.0
        bottom = height - 1.0
        horizontal_target = (right if center[0] < right else 0.0, center[1])
        vertical_target = (center[0], bottom if center[1] < bottom else 0.0)
        diagonal_target = (0.0, 0.0)

        for name, case_target in (
            ("horizontal", horizontal_target),
            ("vertical", vertical_target),
            ("diagonal/corner", diagonal_target),
        ):
            result = line_edge_intersections(width, height, center, case_target)
            _assert_intersections(width, height, center, case_target, result)

        try:
            line_edge_intersections(width, height, center, center)
        except ValueError:
            pass
        else:
            raise AssertionError("Center-equals-target must be rejected.")

        if canvas_to_image_point(0, 0, (10, 20, 101, 51), width, height) is not None:
            raise AssertionError("Canvas letterbox clicks must be ignored.")
        mapped_last_pixel = canvas_to_image_point(
            110,
            70,
            (10, 20, 101, 51),
            width,
            height,
        )
        if mapped_last_pixel != (right, bottom):
            raise AssertionError("Canvas-to-image mapping missed the last source pixel.")

    except (OSError, ValueError, cv2.error, AssertionError) as exc:
        print(f"SELF-TEST FAILED: {exc}", file=sys.stderr)
        return 1

    print(f"SELF-TEST PASSED: {image_path} ({width} x {height} px)")
    print(
        f"Random target ({target[0]:.2f}, {target[1]:.2f}) -> "
        f"A {random_endpoints[0]}, B {random_endpoints[1]}"
    )
    print("Horizontal, vertical, corner, center-target, click-mapping, and OpenCV checks passed.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plan Dragon Duke routes on a base screenshot.")
    parser.add_argument(
        "--self-test",
        metavar="IMAGE",
        help="Run headless geometry and OpenCV checks using IMAGE.",
    )
    args = parser.parse_args(argv)
    if args.self_test:
        return run_self_test(args.self_test)

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        print(f"Could not start the Tkinter window: {exc}", file=sys.stderr)
        return 2

    DragonDukePlanner(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

