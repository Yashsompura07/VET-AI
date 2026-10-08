"""Generate crisp modern PWA icons for VetAI using Pillow."""
import os
from PIL import Image, ImageDraw, ImageFont

FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "frontend"))
os.makedirs(FRONTEND_DIR, exist_ok=True)


def draw_vetai_icon(size: int) -> Image.Image:
    # High-res canvas with supersampling for smooth antialiased curves
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Background rounded rectangle / squircle
    margin = size * 0.04
    radius = size * 0.22
    
    # Deep slate emerald gradient feel: base color #064E3B to #047857
    bg_color = (11, 15, 25, 255) # Deep dark theme background
    draw.rounded_rectangle([margin, margin, size - margin, size - margin], radius=radius, fill=bg_color)
    
    # Inner border glowing emerald
    border_width = max(2, int(size * 0.025))
    draw.rounded_rectangle([margin, margin, size - margin, size - margin], radius=radius, outline=(16, 185, 129, 200), width=border_width)

    # Center shield / veterinary cross
    center_x, center_y = size / 2, size / 2
    
    # Veterinary cross dimensions
    arm_length = size * 0.28
    arm_width = size * 0.12
    r_corner = max(2, int(size * 0.025))
    
    # Horizontal arm of cross
    draw.rounded_rectangle([
        center_x - arm_length, center_y - arm_width,
        center_x + arm_length, center_y + arm_width
    ], radius=r_corner, fill=(16, 185, 129, 255))
    
    # Vertical arm of cross
    draw.rounded_rectangle([
        center_x - arm_width, center_y - arm_length,
        center_x + arm_width, center_y + arm_length
    ], radius=r_corner, fill=(16, 185, 129, 255))
    
    # Center white / light dot or stethoscope heart highlight
    pulse_radius = size * 0.06
    draw.ellipse([
        center_x - pulse_radius, center_y - pulse_radius,
        center_x + pulse_radius, center_y + pulse_radius
    ], fill=(255, 255, 255, 240))
    
    return img


def generate_svg() -> str:
    return """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512">
  <rect width="512" height="512" rx="112" fill="#0B0F19"/>
  <rect x="12" y="12" width="488" height="488" rx="100" fill="none" stroke="#10B981" stroke-width="12" stroke-opacity="0.8"/>
  <rect x="112" y="216" width="288" height="80" rx="16" fill="#10B981"/>
  <rect x="216" y="112" width="80" height="288" rx="16" fill="#10B981"/>
  <circle cx="256" cy="256" r="32" fill="#FFFFFF"/>
</svg>"""


def main():
    icon_192 = draw_vetai_icon(192)
    path_192 = os.path.join(FRONTEND_DIR, "icon-192.png")
    icon_192.save(path_192, format="PNG")
    print(f"Generated {path_192}")

    icon_512 = draw_vetai_icon(512)
    path_512 = os.path.join(FRONTEND_DIR, "icon-512.png")
    icon_512.save(path_512, format="PNG")
    print(f"Generated {path_512}")

    svg_content = generate_svg()
    path_svg = os.path.join(FRONTEND_DIR, "icon.svg")
    with open(path_svg, "w", encoding="utf-8") as f:
        f.write(svg_content)
    print(f"Generated {path_svg}")


if __name__ == "__main__":
    main()
