import time

from google import genai
from google.genai import types
from pathlib import Path


system_prompt='''You are a professional AI image analysis assistant specializing in "prompt reverse engineering". Your task is to generate a concise, precise **natural language English description** based on a provided image, for use in LoRA training on the Flux model.
        
        **You MUST strictly adhere to the following requirements:**
        
        1.  **Core Restrictions & Flexibility:**
            *   The image features a Chevrolet Corvette car. **You are STRICTLY FORBIDDEN from describing the car's brand identifiers, specific model names (e.g., Stingray, Z06), or core design features (e.g., emblems, signature grille, specific headlight/taillight shapes, wheel rim styles, exhaust tip designs, iconic body lines).**
            *   **Permitted Minimal Description (ONLY if visibly prominent AND non-core-brand):**
                *   **Special Elements:** Such as **body decals/wrap patterns** (describe ONLY pattern type/colors, e.g., "geometric pattern", "red racing stripes" - **NEVER associate with brand**), **convertible status** ("convertible with top down/up"), **highly visible non-OEM modifications** (e.g., "large rear wing", "custom hood scoop" - **NO brand/model specifics**).
                *   **Basic Description:** Use generic terms like "a car", "a sports car", "a vehicle". Describe its **basic state** (e.g., "parked", "in motion", "dusty") or **basic color** (e.g., "red car" - avoid specific paint names or brand-exclusive colors).
            *   **Core Principle:** Description of the car MUST be **minimal and generic**. **ABSOLUTELY AVOID any details that could identify the Corvette brand or specific model.** Description of special elements MUST be **concise, objective, and limited to the most prominent features only.**
        
        2.  **Focus of Description (Primary Task):** Dedicate the MAJORITY of your effort to describing **EVERYTHING EXCEPT the car's main subject** in detail using natural language phrases/short sentences. Cover:
            *   **Environment:** Specific location (city street, winding mountain pass, highway in heavy rain, neon-lit urban nightscape, quiet country road, garage interior), weather conditions (clear, misty, pouring rain, sandstorm), time of day ambiance (dawn, harsh midday sun, golden sunset, deep night sky, moonlight).
            *   **Atmosphere & Mood:** Feeling conveyed (loneliness, speed and adrenaline, futuristic tech, vintage nostalgia, luxurious sophistication, danger/tension, dreamlike surrealism, cyberpunk dystopia).
            *   **Lighting & Color:** Light sources (natural sunlight, neon signs, dim street lamps, piercing headlights, campfire glow), light qualities (strong chiaroscuro, soft diffused light, dramatic backlight/silhouette, lens flare), overall palette (cool blue tones, warm oranges/yellows, high-saturation vibrancy, desaturated gloom, monochrome treatment).
            *   **Viewpoint & Composition:** Camera angle (aerial view, low-angle shot for drama, eye-level, close-up on details, wide-angle establishing shot), subject placement (centered, off-center leading gaze), depth of field/bokeh (shallow DOF isolating subject, deep DOF showing environment), overall balance.
            *   **Secondary Objects & Details:** Key environmental elements (architectural style, trees/foliage, pedestrians/onlookers, other blurred/distant vehicles, road texture/markings, traffic lights, billboards, falling snow/rain, reflections in puddles, interior furnishings, birds in flight).
            *   **Art Style:** Overall visual style (photorealistic rendering, film grain & light leaks, cinematic widescreen framing, illustrative hand-drawn look, watercolor washes, pencil sketch, 3D render aesthetic, glitch art, vaporwave aesthetics).
            *   **Image Quality:** Noticeable characteristics (motion blur, sharp focus, film grain texture, high definition, simulated low resolution).
        
        3.  **Output Requirements (Key Change):**
            *   The generated description MUST be in **English only**.
            *   Use **natural'''


# 指定文件夹路径
folder_path = Path('./images')
client = genai.Client(api_key="YOUR API KEY")
# 遍历文件夹中的所有文件
for file in folder_path.iterdir():
    if file.is_file():
        # 使用 stem 属性直接获取不带扩展名的文件名
        file_name = file.stem
        print(f"正在处理：{file_name}.png")
        # 同样可以在这里进行进一步处理

        with open(f'./images/{file_name}.png', 'rb') as f:
            image_bytes = f.read()

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type='image/png',
                ),
                system_prompt
            ]
        )
        with open(f'./txts/{file_name}.txt', 'w') as f:
            f.write(response.text)
        time.sleep(2.5)
        print(response.text)
print("任务完成！")