# Beta V0.72 - fine version
import numpy as np
from tqdm import trange
from PIL import Image, ImageSequence, ImageDraw
import math

import modules.scripts as scripts
import gradio as gr

from modules import processing, shared, sd_samplers, images
from modules.processing import Processed
from modules.sd_samplers import samplers
from modules.shared import opts, cmd_opts, state
from modules import deepbooru


class Script(scripts.Script):
    def title(self):
        return "Multi-frame Video - V0.72-beta (fine version)"

    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):   
    
        with gr.Row(variant="panel"):
        
            loopback_source = gr.Dropdown(label="Loopback Source", choices=["FirstGen", "InputFrame", "PreviousFrame", "CNetFrame" ], value="FirstGen")
            
            third_frame_image = gr.Dropdown(label="Third Frame", choices=["None", "FirstGen", "GuideImg", "Historical"], value="FirstGen")
            
            append_interrogation = gr.Dropdown(label="Append prompt", choices=["None", "CLIP",  "CLIP Historical", "CLIP CNet", "DeepBooru", "DeepBooru Historical","DeepBooru CNet"], value="None")
            
            use_nth_frame = gr.Number(label="Use every Nth frame", value="1")
    

        with gr.Row(variant="compact"):
            
            render_grid = gr.Checkbox(label="Render grid", value=False, elem_id=self.elem_id("render_grid"))   
            
            grid_rows = gr.Number(label="Rows in grid", value="1")
        
        with gr.Row(variant="panel"):
            first_denoise = gr.Slider(minimum=0, maximum=1, step=0.05, label='Initial Denoising', value=0, elem_id=self.elem_id("first_denoise"))  

            color_correction_enabled = gr.Checkbox(label="Color Correction", value=True, elem_id=self.elem_id("color_correction_enabled"))
            
            unfreeze_seed = gr.Checkbox(label="Unfreeze Seed", value=False, elem_id=self.elem_id("unfreeze_seed"))            
        
        with gr.Row(variant="panel"):        
            reference_imgs = gr.File(file_count="multiple", file_types = ['.png','.jpg','.jpeg'], label="Upload Guide Frames", show_label=True, live=True)
            
        with gr.Accordion(label="INFO!", open=False):
            gr.HTML(value="<p style='margin-top: 10rem, margin-bottom: 10rem'>This is a modified script originally written by <a href='https://xanthius.itch.io/multi-frame-rendering-for-stablediffusion'>xanthius</a> (click the link for details and buy him a coffee!)</p><p>I cleaned up, changed the defaults and sorted the UI. The default values are set to what works best for me in most cases.</p><p>I also added some quality-of-life features:<br><ul><li>Use every Nth frame: skip guide frames (for preview or ebsynth)</li><li>Render grid: enable to render the grid</li><li>Rows in grid: how many horizontal rows the grid should have</li> <li>Fixed file upload</li></ul><li>added more interrogation options</li></p>")

        return [append_interrogation, reference_imgs, first_denoise, third_frame_image, color_correction_enabled, unfreeze_seed, render_grid,grid_rows, use_nth_frame, loopback_source]

    def run(self, p, append_interrogation, reference_imgs, first_denoise, third_frame_image, color_correction_enabled, unfreeze_seed, render_grid, grid_rows, use_nth_frame, loopback_source):
        freeze_seed = not unfreeze_seed
        use_nth_frame = int (use_nth_frame)
        grid_rows = int (grid_rows)
        
        loops = math.floor (len(reference_imgs) / use_nth_frame)

        processing.fix_seed(p)
        # batch_count = math.floor (p.n_iter / use_nth_frame)
        batch_count = p.n_iter
        
        shared.log.info(f"p.n_iter={p.n_iter}  use_nth_frame={use_nth_frame} batch_count={batch_count} loops={loops}")
        
        p.batch_size = 1
        p.n_iter = 1

        output_images, info = None, None
        initial_seed = None
        initial_info = None

        initial_width = p.width
        initial_img = p.init_images[0]

        grids = []
        all_images = []
        original_init_image = p.init_images
        original_prompt = p.prompt
        original_denoise = p.denoising_strength
        state.job_count = loops * batch_count

        initial_color_corrections = [processing.setup_color_correction(p.init_images[0])]

        for n in range(batch_count):
            history = []
            frames = []
            third_image = None
            third_image_index = 0
            frame_color_correction = None

            # Reset to original init image at the start of each batch
            p.init_images = original_init_image
            p.width = initial_width

            for i in range(loops):
                p.n_iter = 1
                p.batch_size = 1
                p.do_not_save_grid = True
                p.control_net_input_image = Image.open(reference_imgs[i*use_nth_frame].name).convert("RGB").resize((initial_width, p.height), Image.ANTIALIAS)

                if(i > 0):
                    loopback_image = p.init_images[0]
                    if loopback_source == "InputFrame":
                        loopback_image = p.control_net_input_image
                    elif loopback_source == "FirstGen":
                        loopback_image = history[0]


                    if third_frame_image != "None" and i > 1:
                        p.width = initial_width * 3
                        img = Image.new("RGB", (initial_width*3, p.height))
                        img.paste(p.init_images[0], (0, 0))
                        # img.paste(p.init_images[0], (initial_width, 0))
                        img.paste(loopback_image, (initial_width, 0))
                        img.paste(third_image, (initial_width*2, 0))
                        p.init_images = [img]
                        if color_correction_enabled:
                            p.color_corrections = [processing.setup_color_correction(img)]

                        msk = Image.new("RGB", (initial_width*3, p.height))
                        msk.paste(Image.open(reference_imgs[i-1].name).convert("RGB").resize((initial_width, p.height), Image.ANTIALIAS), (0, 0))
                        msk.paste(p.control_net_input_image, (initial_width, 0))
                        msk.paste(Image.open(reference_imgs[third_image_index].name).convert("RGB").resize((initial_width, p.height), Image.ANTIALIAS), (initial_width*2, 0))
                        p.control_net_input_image = msk

                        latent_mask = Image.new("RGB", (initial_width*3, p.height), "black")
                        latent_draw = ImageDraw.Draw(latent_mask)
                        latent_draw.rectangle((initial_width,0,initial_width*2,p.height), fill="white")
                        p.image_mask = latent_mask
                        p.denoising_strength = original_denoise
                    else:
                        p.width = initial_width * 2
                        img = Image.new("RGB", (initial_width*2, p.height))
                        img.paste(p.init_images[0], (0, 0))
                        # img.paste(p.init_images[0], (initial_width, 0))
                        img.paste(loopback_image, (initial_width, 0))
                        p.init_images = [img]
                        if color_correction_enabled:
                            p.color_corrections = [processing.setup_color_correction(img)]

                        msk = Image.new("RGB", (initial_width*2, p.height))
                        msk.paste(Image.open(reference_imgs[i-1].name).convert("RGB").resize((initial_width, p.height), Image.ANTIALIAS), (0, 0))
                        msk.paste(p.control_net_input_image, (initial_width, 0))
                        p.control_net_input_image = msk
                        frames.append(msk)

                        # latent_mask = Image.new("RGB", (initial_width*2, p.height), "white")
                        # latent_draw = ImageDraw.Draw(latent_mask)
                        # latent_draw.rectangle((0,0,initial_width,p.height), fill="black")
                        latent_mask = Image.new("RGB", (initial_width*2, p.height), "black")
                        latent_draw = ImageDraw.Draw(latent_mask)
                        latent_draw.rectangle((initial_width,0,initial_width*2,p.height), fill="white")

                        # p.latent_mask = latent_mask
                        p.image_mask = latent_mask
                        p.denoising_strength = original_denoise
                else:
                    latent_mask = Image.new("RGB", (initial_width, p.height), "white")
                    # p.latent_mask = latent_mask
                    p.image_mask = latent_mask
                    p.denoising_strength = first_denoise
                    p.control_net_input_image = p.control_net_input_image.resize((initial_width, p.height))
                    frames.append(p.control_net_input_image)
                   
                processed = processing.process_images(p) 

                if append_interrogation != "None":
                    p.prompt = original_prompt + ", " if original_prompt != "" else ""
                    if append_interrogation == "CLIP":
                        p.prompt += shared.interrogator.interrogate(p.init_images[0])
                    elif append_interrogation == "DeepBooru":
                        p.prompt += deepbooru.model.tag(p.init_images[0])
                    elif append_interrogation == "CLIP Historical":
                        p.prompt += shared.interrogator.interrogate(processed.images[0].crop((0, 0, initial_width, p.height)))
                    elif append_interrogation == "DeepBooru Historical":
                        p.prompt += deepbooru.model.tag(processed.images[0].crop((0, 0, initial_width, p.height)))
                    elif append_interrogation == "CLIP CNet":
                        p.prompt += shared.interrogator.interrogate(p.control_net_input_image)
                    elif append_interrogation == "DeepBooru CNet":
                        p.prompt += deepbooru.model.tag(p.control_net_input_image)

                state.job = f"Iteration {i + 1}/{loops}, batch {n + 1}/{batch_count}" 

                if initial_seed is None:
                    initial_seed = processed.seed
                    initial_info = processed.info

                init_img = processed.images[0]
                if(i > 0):
                    init_img = init_img.crop((initial_width, 0, initial_width*2, p.height))

                if third_frame_image != "None":
                    if third_frame_image == "FirstGen" and i == 0:
                        third_image = init_img
                        third_image_index = 0
                    elif third_frame_image == "GuideImg" and i == 0:
                        third_image = original_init_image[0]
                        third_image_index = 0
                    elif third_frame_image == "Historical":
                        third_image = processed.images[0].crop((0, 0, initial_width, p.height))
                        third_image_index = (i-1)

                p.init_images = [init_img]
                if(freeze_seed):
                    p.seed = processed.seed
                else:
                    p.seed = processed.seed + 1

                history.append(init_img)
                if opts.samples_save:
                    images.save_image(init_img, p.outpath_samples, "Frame", p.seed, p.prompt, opts.grid_format, info=info, short_filename=not opts.grid_extended_filename, grid=True, p=p)

                frames.append(processed.images[0])

            if render_grid:
                nRows = grid_rows
                grid = images.image_grid(history, rows=nRows)
                if opts.grid_save:
                    images.save_image(grid, p.outpath_grids, "grid", initial_seed, p.prompt, opts.grid_format, info=info, short_filename=not opts.grid_extended_filename, grid=True, p=p)
                grids.append(grid)
                
            # all_images += history + frames
            all_images += history

            p.seed = p.seed+1

        if opts.return_grid:
            all_images = grids + all_images

        processed = Processed(p, all_images, initial_seed, initial_info)

        return processed
