# taj

深度学习课程项目，因为文件太大和某些不便原因，只上传关键部分，不方便完全开源。（实验报告附件有运行全流程过程截图）

`df_taj` 为数据集，使用时进行解压即可。视频为 472 个，时长大于且接近 6 秒，分辨率为 988*720。

`gradio_web_demo.py` 文件为推理文件，里面进行了 prompt 优化，为用户的输入进行优化。

指令微调 demo 如下：

```json
{
    "role": "system",
    "content": sys_prompt
},
{
    "role": "user",
    "content": 'Generate a detailed animation description for the input: or modify an earlier caption for the user input : "Tom cat is watching the night sky"'
},
{
    "role": "assistant",
    "content": "In the style of Tom cat and Jerry mouse animation, Tom, the mischievous gray cat, is captured in a moment of intense focus and determination. He stands on his hind legs in a room with a large window that offers a view of the starry night sky. His eyes are wide open, reflecting the light from the window, and his ears are perked up, indicating his heightened alertness. In his right paw, he holds a small, shiny object, which he seems to be examining closely. His left paw is slightly raised, as if he's about to make a sudden move. The room has a futuristic design, with sleek lines and a minimalist aesthetic. The overall atmosphere is one of suspense and anticipation, as if Tom is about to spring into action at any moment."
}
