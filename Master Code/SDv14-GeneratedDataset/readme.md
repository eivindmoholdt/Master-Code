This dataset is automatically generated using Stable Diffusion text-to-image model, for research purposes.
The images are generated based on captions from the COSMOS dataset: https://github.com/sanonymous22/COSMOS
The images are generated using CompVis Stable Diffusion model v1.4 found here: https://huggingface.co/CompVis/stable-diffusion-v1-4

with config:
experimental_pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", revision="fp16", torch_dtype=torch.float16, use_auth_token=True) 
experimental_pipe = experimental_pipe.to("cuda")
