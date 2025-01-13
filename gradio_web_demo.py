import os
import tempfile
import threading
import time

import gradio as gr
import numpy as np
import torch
from diffusers import CogVideoXPipeline
from datetime import datetime, timedelta
from openai import OpenAI
import imageio
import moviepy.editor as mp
from typing import List, Union
import PIL

dtype = torch.bfloat16
device = "cuda" if torch.cuda.is_available() else "cpu"
# pipe = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-2b", torch_dtype=dtype)
# pipe.enable_model_cpu_offload()
# 加载基础模型和微调后的 LoRA 权重
pipe = CogVideoXPipeline.from_pretrained(
    "THUDM/CogVideoX-2b", 
    torch_dtype=dtype,
    adapter_path="/root/autodl-tmp/cogvideox-lora1/pytorch_lora_weights.safetensors"  # 指定 LoRA 权重路径
)
pipe.enable_model_cpu_offload()

# 启用 LoRA 适配器
# pipe.load_adapter("/root/autodl-tmp/cogvideox-lora1")
# pipe.enable_model_cpu_offload()
sys_prompt = """You are part of a team of bots that creates animation videos. You work with an assistant bot that will draw anything you say in square brackets.

For example , outputting " a beautiful morning in the woods with the sun peaking through the trees " will trigger your partner bot to output an video of a forest morning , as described. You will be prompted by people looking to create detailed , amazing videos. The way to accomplish this is to take their short prompts and make them extremely detailed and descriptive.
There are a few rules to follow:
You should create detailed, imaginative animation descriptions in the style of 'Tom cat and Jerry mouse'.Emphasis is placed on describing the characteristics and actions of the character, describing the environment and atmosphere, as well as detailing and sensory experiences.

You need to generate a description that can guide the model to generate animations

You will only ever output a single video description per user request.

When modifications are requested , you should not simply make the description longer . You should refactor the entire description to integrate the suggestions.
Other times the user will not want modifications , but instead want a new image . In this case , you should ignore your previous conversation with the user.

Video descriptions must have the same num of words as examples below. Extra words will be ignored.
"""


def export_to_video_imageio(
    video_frames: Union[List[np.ndarray], List[PIL.Image.Image]], output_video_path: str = None, fps: int = 8
) -> str:
    """
    Export the video frames to a video file using imageio lib to Avoid "green screen" issue (for example CogVideoX)
    """
    if output_video_path is None:
        output_video_path = tempfile.NamedTemporaryFile(suffix=".mp4").name

    if isinstance(video_frames[0], PIL.Image.Image):
        video_frames = [np.array(frame) for frame in video_frames]

    with imageio.get_writer(output_video_path, fps=fps) as writer:
        for frame in video_frames:
            writer.append_data(frame)

    return output_video_path


def convert_prompt(prompt: str, retry_times: int = 3) -> str:
    # if not os.environ.get("OPENAI_API_KEY"):
    #     return prompt
    client = OpenAI(api_key="68463c7ea28f41b7872736634062fbf9.ZVzzGKP2devtbxoG", base_url="https://open.bigmodel.cn/api/paas/v4/")
    text = prompt.strip()

    for i in range(retry_times):
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": sys_prompt},
                {
                    "role": "user",
                    "content": 'Generate a detailed animation description for the input: or modify an earlier caption for the user input : "Tom cat is watching the night sky"',
                },
                {
                    "role": "assistant",
                    "content":"In the style of Tom cat and Jerry mouse animation,Tom, the mischievous gray cat, is captured in a moment of intense focus and determination. He stands on his hind legs in a room with a large window that offers a view of the starry night sky. His eyes are wide open, reflecting the light from the window, and his ears are perked up, indicating his heightened alertness. In his right paw, he holds a small, shiny object, which he seems to be examining closely. His left paw is slightly raised, as if he's about to make a sudden move. The room has a futuristic design, with sleek lines and a minimalist aesthetic. The overall atmosphere is one of suspense and anticipation, as if Tom is about to spring into action at any moment.",
                },
                {
                    "role": "user",
                    "content":'Generate a detailed animation description for the input: or modify an earlier caption for the user input : "Jerry mouse found a large cheese in the kitchen."',
                },
                {
                    "role": "assistant",
                    "content":"In the style of Tom cat and Jerry mouse animation,Jerry, the clever little brown mouse, is seen darting across a cluttered kitchen countertop. His tiny paws move with lightning speed as he navigates through a maze of utensils, jars, and crumbs. His eyes gleam with mischief, and his whiskers twitch with excitement as he spots a large wedge of cheese sitting precariously on the edge of a cutting board. Jerry's tail flicks behind him like a rudder, helping him maintain balance as he leaps over a rolling spoon. The kitchen is bathed in warm, golden light from the setting sun, casting long shadows that add to the sense of urgency and adventure. The scene is filled with playful tension, as Jerry's quick thinking and agility are put to the test in his quest for the prized cheese.",
                },
                {
                    "role": "user",
                    "content": 'Generate a detailed animation description for the input: or modify an earlier caption for the user input : "Tom cat found that Jerry get a cheese and Tom try to catch Jerry."',
                },
                {
                    "role": "assistant",
                    "content": "In the style of Tom cat and Jerry mouse animation,In a lively farmhouse kitchen bathed in warm sunlight, Tom, the resolute gray cat with a sleek coat and piercing green eyes, crouches stealthily behind an old wooden barrel, his tail flicking with eager anticipation. Across from him, Jerry, the dexterous brown mouse with a cheeky glint in his eyes, balances precariously on the flour-dusted edge of a countertop. Jerry clutches a golden sliver of cheese in one tiny paw and wields a spoon like a sword in the other, his whiskers quivering with a blend of playful mischief and vigilant wariness as he keeps a watchful eye on Tom's every move. The kitchen swirls with activity: a pot of soup bubbles menacingly on the stove, a forgotten rolling pin lies on the floor amidst a scattered flour cloud, and a trail of delicate paw prints meanders from the pantry to the countertop battlefield. In a flash, Tom pounces with a thunderous leap, claws extended for the kill, but the swift Jerry is one step ahead, plunging into a tiny teacup that whirls across the counter in a comical dance. This animated scene, brimming with slapstick humor and a heart-racing.",
                },
                {
                    "role": "user",
                    "content": f'Generate a detailed animation description for the input: or modify an earlier caption for the user input: "{text}"',
                },
            ],
            model="glm-4-0520",
            temperature=0.01,
            top_p=0.7,
            stream=False,
            max_tokens=250,
        )
        if response.choices:
            return response.choices[0].message.content
    return prompt


def infer(prompt: str, num_inference_steps: int, guidance_scale: float, progress=gr.Progress(track_tqdm=True)):
    torch.cuda.empty_cache()

    prompt_embeds, _ = pipe.encode_prompt(
        prompt=prompt,
        negative_prompt=None,
        do_classifier_free_guidance=True,
        num_videos_per_prompt=1,
        max_sequence_length=226,
        device=device,
        dtype=dtype,
    )

    video = pipe(
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=torch.zeros_like(prompt_embeds),
    ).frames[0]

    return video


def save_video(tensor):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = f"./output/{timestamp}.mp4"
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    export_to_video_imageio(tensor[1:], video_path)
    return video_path


def convert_to_gif(video_path):
    clip = mp.VideoFileClip(video_path)
    clip = clip.set_fps(8)
    clip = clip.resize(height=240)
    gif_path = video_path.replace(".mp4", ".gif")
    clip.write_gif(gif_path, fps=8)
    return gif_path


def delete_old_files():
    while True:
        now = datetime.now()
        cutoff = now - timedelta(minutes=10)
        output_dir = "./output"
        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            if os.path.isfile(file_path):
                file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                if file_mtime < cutoff:
                    os.remove(file_path)
        time.sleep(600)  # Sleep for 10 minutes


threading.Thread(target=delete_old_files, daemon=True).start()

with gr.Blocks() as demo:
    gr.Markdown("""
           <div style="text-align: center; font-size: 32px; font-weight: bold; margin-bottom: 20px;">
               CogVideoX-2B Huggingface Space🤗
           </div>
           <div style="text-align: center;">
               <a href="https://huggingface.co/THUDM/CogVideoX-2b">🤗 Model Hub</a> |
               <a href="https://github.com/THUDM/CogVideo">🌐 Github</a> 
           </div>

           <div style="text-align: center; font-size: 15px; font-weight: bold; color: red; margin-bottom: 20px;">
            ⚠️ This demo is for academic research and experiential use only. 
            Users should strictly adhere to local laws and ethics.
            </div>
           """)
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Prompt (Less than 200 Words)", placeholder="Enter your prompt here", lines=5)
            with gr.Row():
                gr.Markdown(
                    "✨Upon pressing the enhanced prompt button, we will use [GLM-4 Model](https://github.com/THUDM/GLM-4) to polish the prompt and overwrite the original one."
                )
                enhance_button = gr.Button("✨ Enhance Prompt(Optional)")

            with gr.Column():
                gr.Markdown(
                    "**Optional Parameters** (default values are recommended)<br>"
                    "Turn Inference Steps larger if you want more detailed video, but it will be slower.<br>"
                    "50 steps are recommended for most cases. will cause 120 seconds for inference.<br>"
                )
                with gr.Row():
                    num_inference_steps = gr.Number(label="Inference Steps", value=50)
                    guidance_scale = gr.Number(label="Guidance Scale", value=6.0)
                generate_button = gr.Button("🎬 Generate Video")

        with gr.Column():
            video_output = gr.Video(label="CogVideoX Generate Video", width=720, height=480)
            with gr.Row():
                download_video_button = gr.File(label="📥 Download Video", visible=False)
                download_gif_button = gr.File(label="📥 Download GIF", visible=False)

    gr.Markdown("""
        <table border="1" style="width: 100%; text-align: left; margin-top: 20px;">
            <tr>
                <th>Prompt</th>
                <th>Video URL</th>
                <th>Inference Steps</th>
                <th>Guidance Scale</th>
            </tr>
            <tr>
                <td>A detailed wooden toy ship with intricately carved masts and sails is seen gliding smoothly over a plush, blue carpet that mimics the waves of the sea. The ship's hull is painted a rich brown, with tiny windows. The carpet, soft and textured, provides a perfect backdrop, resembling an oceanic expanse. Surrounding the ship are various other toys and children's items, hinting at a playful environment. The scene captures the innocence and imagination of childhood, with the toy ship's journey symbolizing endless adventures in a whimsical, indoor setting.</td>
                <td><a href="https://github.com/THUDM/CogVideo/raw/main/resources/videos/1.mp4">Video 1</a></td>
                <td>50</td>
                <td>6</td>
            </tr>
            <tr>
                <td>The camera follows behind a white vintage SUV with a black roof rack as it speeds up a steep dirt road surrounded by pine trees on a steep mountain slope, dust kicks up from it’s tires, the sunlight shines on the SUV as it speeds along the dirt road, casting a warm glow over the scene. The dirt road curves gently into the distance, with no other cars or vehicles in sight. The trees on either side of the road are redwoods, with patches of greenery scattered throughout. The car is seen from the rear following the curve with ease, making it seem as if it is on a rugged drive through the rugged terrain. The dirt road itself is surrounded by steep hills and mountains, with a clear blue sky above with wispy clouds.</td>
                <td><a href="https://github.com/THUDM/CogVideo/raw/main/resources/videos/2.mp4">Video 2</a></td>
                <td>50</td>
                <td>6</td>
            </tr>
            <tr>
                <td>A street artist, clad in a worn-out denim jacket and a colorful bandana, stands before a vast concrete wall in the heart, holding a can of spray paint, spray-painting a colorful bird on a mottled wall.</td>
                <td><a href="https://github.com/THUDM/CogVideo/raw/main/resources/videos/3.mp4">Video 3</a></td>
                <td>50</td>
                <td>6</td>
            </tr>
            <tr>
                <td>In the haunting backdrop of a war-torn city, where ruins and crumbled walls tell a story of devastation, a poignant close-up frames a young girl. Her face is smudged with ash, a silent testament to the chaos around her. Her eyes glistening with a mix of sorrow and resilience, capturing the raw emotion of a world that has lost its innocence to the ravages of conflict.</td>
                <td><a href="https://github.com/THUDM/CogVideo/raw/main/resources/videos/4.mp4">Video 4</a></td>
                <td>50</td>
                <td>6</td>
            </tr>
        </table>
    """)

    def generate(prompt, num_inference_steps, guidance_scale, progress=gr.Progress(track_tqdm=True)):
        tensor = infer(prompt, num_inference_steps, guidance_scale, progress=progress)
        video_path = save_video(tensor)
        video_update = gr.update(visible=True, value=video_path)
        gif_path = convert_to_gif(video_path)
        gif_update = gr.update(visible=True, value=gif_path)

        return video_path, video_update, gif_update

    def enhance_prompt_func(prompt):
        return convert_prompt(prompt, retry_times=1)

    generate_button.click(
        generate,
        inputs=[prompt, num_inference_steps, guidance_scale],
        outputs=[video_output, download_video_button, download_gif_button],
    )

    enhance_button.click(enhance_prompt_func, inputs=[prompt], outputs=[prompt])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=6006, share=True)
