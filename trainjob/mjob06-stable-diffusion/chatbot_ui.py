# ============================================
# Stable Diffusion - Chatbot UI
# Streamlit-based Image Generation Interface
# ============================================

import streamlit as st
import requests
import base64
from io import BytesIO
from PIL import Image
import os

# Configuration
CHATBOT_URL = os.environ.get("CHATBOT_URL", "http://localhost:8080")

st.set_page_config(
    page_title="Stable Diffusion Image Generator",
    page_icon="🎨",
    layout="wide"
)

st.title("🎨 Stable Diffusion Image Generator")
st.markdown("텍스트로 이미지를 생성하는 AI 챗봇입니다.")

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")

    # Image settings
    st.subheader("Image Settings")
    width = st.select_slider("Width", options=[256, 384, 512, 640, 768], value=512)
    height = st.select_slider("Height", options=[256, 384, 512, 640, 768], value=512)
    steps = st.slider("Inference Steps", 10, 100, 30)
    guidance = st.slider("Guidance Scale", 1.0, 20.0, 7.5)

    # Style presets
    st.subheader("Style Presets")
    style = st.selectbox(
        "Select Style",
        options=[
            "None",
            "Realistic",
            "Anime",
            "Oil Painting",
            "Watercolor",
            "Cyberpunk",
            "Fantasy",
            "Minimalist",
            "3D Render"
        ]
    )

    # Advanced options
    st.subheader("Advanced")
    use_llm_optimize = st.checkbox("Use LLM to optimize prompt", value=True)
    seed = st.number_input("Seed (-1 for random)", value=-1, min_value=-1)

    # Service status
    st.subheader("Service Status")
    try:
        health = requests.get(f"{CHATBOT_URL}/health", timeout=5).json()
        st.success(f"Chatbot: ✅ Running")
        st.info(f"SD Server: {health.get('sd_server', 'unknown')}")
        st.info(f"Ollama: {health.get('ollama', 'unknown')}")
    except:
        st.error("Chatbot: ❌ Disconnected")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "generated_images" not in st.session_state:
    st.session_state.generated_images = []

# Create tabs
tab1, tab2, tab3 = st.tabs(["💬 Chat", "🖼️ Gallery", "📖 Help"])

with tab1:
    # Display chat messages
    chat_container = st.container()

    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                if "image" in msg and msg["image"]:
                    st.image(msg["image"], caption=msg.get("prompt", "Generated Image"))

    # Chat input
    prompt = st.chat_input("이미지를 설명해주세요 (예: 'a cat sitting on a rainbow')")

    if prompt:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.write(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🎨 이미지 생성 중..."):
                try:
                    # Prepare request
                    image_settings = {
                        "width": width,
                        "height": height,
                        "num_inference_steps": steps,
                        "guidance_scale": guidance
                    }

                    if seed != -1:
                        image_settings["seed"] = seed

                    # Add style to prompt if selected
                    style_suffix = ""
                    if style != "None":
                        style_map = {
                            "Realistic": ", photorealistic, 8k, professional photography",
                            "Anime": ", anime style, vibrant colors, detailed",
                            "Oil Painting": ", oil painting, classical art style, rich textures",
                            "Watercolor": ", watercolor painting, soft colors, artistic",
                            "Cyberpunk": ", cyberpunk style, neon lights, futuristic",
                            "Fantasy": ", fantasy art, magical, epic composition",
                            "Minimalist": ", minimalist design, clean lines, simple",
                            "3D Render": ", 3D render, octane render, highly detailed"
                        }
                        style_suffix = style_map.get(style, "")

                    full_prompt = prompt + style_suffix

                    # Call API
                    if use_llm_optimize:
                        response = requests.post(
                            f"{CHATBOT_URL}/chat",
                            json={
                                "message": full_prompt,
                                "generate_image": True,
                                "image_settings": image_settings
                            },
                            timeout=600
                        )
                    else:
                        response = requests.post(
                            f"{CHATBOT_URL}/generate-direct",
                            json={
                                "message": full_prompt,
                                "image_settings": image_settings
                            },
                            timeout=600
                        )

                    if response.status_code == 200:
                        data = response.json()

                        # Display response text
                        if "response" in data:
                            st.write(data["response"])

                        # Display image
                        if data.get("image_base64"):
                            image_data = base64.b64decode(data["image_base64"])
                            image = Image.open(BytesIO(image_data))
                            st.image(image, caption=f"Seed: {data.get('seed', 'N/A')}")

                            # Save to gallery
                            st.session_state.generated_images.append({
                                "image": image,
                                "prompt": data.get("prompt_used", prompt),
                                "seed": data.get("seed")
                            })

                            # Save message with image
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": data.get("response", "이미지가 생성되었습니다!"),
                                "image": image,
                                "prompt": data.get("prompt_used", prompt)
                            })
                        else:
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": data.get("response", "응답을 받지 못했습니다.")
                            })
                    else:
                        st.error(f"Error: {response.status_code}")
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"오류가 발생했습니다: {response.status_code}"
                        })

                except requests.exceptions.Timeout:
                    st.error("요청 시간이 초과되었습니다. 다시 시도해주세요.")
                except Exception as e:
                    st.error(f"오류: {str(e)}")

with tab2:
    st.header("🖼️ Generated Images Gallery")

    if st.session_state.generated_images:
        cols = st.columns(3)
        for idx, item in enumerate(reversed(st.session_state.generated_images)):
            with cols[idx % 3]:
                st.image(item["image"], caption=f"Seed: {item.get('seed', 'N/A')}")
                st.caption(item.get("prompt", "")[:100] + "...")

                # Download button
                buf = BytesIO()
                item["image"].save(buf, format="PNG")
                st.download_button(
                    label="📥 Download",
                    data=buf.getvalue(),
                    file_name=f"generated_{item.get('seed', 'image')}.png",
                    mime="image/png",
                    key=f"download_{idx}"
                )
    else:
        st.info("아직 생성된 이미지가 없습니다. Chat 탭에서 이미지를 생성해보세요!")

with tab3:
    st.header("📖 사용 가이드")

    st.markdown("""
    ## Stable Diffusion 이미지 생성기

    ### 기본 사용법
    1. **Chat 탭**에서 원하는 이미지를 텍스트로 설명합니다
    2. AI가 프롬프트를 최적화하고 이미지를 생성합니다
    3. 생성된 이미지는 **Gallery 탭**에 저장됩니다

    ### 좋은 프롬프트 작성 팁

    #### 기본 구조
    ```
    [주제] + [스타일] + [조명/분위기] + [품질 키워드]
    ```

    #### 예시
    - "a cute cat wearing a wizard hat, digital art, magical atmosphere, highly detailed"
    - "sunset over mountain lake, landscape photography, golden hour, 8k resolution"
    - "cyberpunk city street, neon lights, rain, cinematic lighting, detailed"

    ### 스타일 프리셋
    | 스타일 | 특징 |
    |--------|------|
    | Realistic | 사진처럼 사실적인 이미지 |
    | Anime | 일본 애니메이션 스타일 |
    | Oil Painting | 유화 느낌의 클래식한 스타일 |
    | Cyberpunk | 네온과 미래적 분위기 |
    | Fantasy | 판타지/마법 느낌 |

    ### 설정 옵션
    - **Width/Height**: 이미지 크기 (512x512 권장)
    - **Steps**: 높을수록 품질↑, 시간↑ (30 권장)
    - **Guidance Scale**: 프롬프트 반영 강도 (7.5 권장)
    - **Seed**: 같은 시드 = 같은 결과 (재현 가능)

    ### Fine-tuning (학습)
    사용자만의 스타일이나 캐릭터를 학습시키려면:
    1. 학습 이미지를 `/mnt/storage/data/train` 폴더에 업로드
    2. 각 이미지에 대한 설명을 `.txt` 파일로 작성
    3. TrainJob 실행으로 LoRA 학습

    예시 데이터 구조:
    ```
    /mnt/storage/data/train/
    ├── my_character_01.png
    ├── my_character_01.txt  # "a photo of sks character"
    ├── my_character_02.png
    └── my_character_02.txt
    ```
    """)
