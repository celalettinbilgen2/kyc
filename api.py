from distutils.log import debug 
from fileinput import filename 
from flask import *
import os
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from PIL import Image
from pathlib import Path
import sys

app = Flask(__name__) 

# Toggle to switch between full response and extracted description
OUTPUT_FULL_RESPONSE = False

# Ensure we're using the MPS device if available
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda")

def load_model_and_tools():
    """Load the Qwen2-VL model, tokenizer, and processor for Apple Silicon."""
    model_name = "Qwen/Qwen2-VL-7B-Instruct"
    try:
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto").to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        processor = AutoProcessor.from_pretrained(model_name)
        return model, tokenizer, processor
    except Exception as e:
        print(f"Failed to load the model: {e}")
        sys.exit(1)

def resize_image(image, max_height, max_width):
    """Resize the image only if it exceeds the specified dimensions."""
    original_width, original_height = image.size
    
    # Check if resizing is needed
    if original_width > max_width or original_height > max_height:
        # Calculate the new size maintaining the aspect ratio
        aspect_ratio = original_width / original_height
        if original_width > original_height:
            new_width = max_width
            new_height = int(max_width / aspect_ratio)
        else:
            new_height = max_height
            new_width = int(max_height * aspect_ratio)
        
        # Resize the image using LANCZOS for high-quality downscaling
        return image.resize((new_width, new_height), Image.LANCZOS)
    else:
        return image

def _gc():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

response = "None"

def process_image(image_path: Path, model, tokenizer, processor, prompt):
    """Process the image and generate a description using the MPS device if available."""
    try:
        with Image.open(image_path) as img:
            # Resize the image if necessary
            max_height = 1260
            max_width = 1260
            img = resize_image(img, max_height, max_width)
            
            messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[img], padding=True, return_tensors="pt").to(device)
            
            # Generate outputs
            outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.5, repetition_penalty=1.1)
            
            # Decode the outputs using tokenizer
            response_ids = outputs[0]
            response_text = tokenizer.decode(response_ids, skip_special_tokens=False)
            
            if OUTPUT_FULL_RESPONSE:
                return f"\n{response_text}" 
                #print(f"\n{response_text}")
            else:
                response = response_text.split('<|im_start|>assistant')[-1].split('<|im_end|>')[0].strip()
                return f"\n{response}"
                #print(f"\n{response}")
    except FileNotFoundError:
        print(f"Error: The image file '{image_path}' was not found.")
    except Image.UnidentifiedImageError:
        print(f"Error: The file '{image_path}' is not a valid image file or is corrupted.")
    except torch.cuda.OutOfMemoryError:
        print("Error: Ran out of GPU memory. Try using a smaller image or freeing up GPU resources.")
    except ValueError as ve:
        print(f"Error: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred while processing '{image_path}': {e}")
        print("Please check your input and try again.")

model, tokenizer, processor = load_model_and_tools()

@app.route('/vlm-upload', methods = ['POST']) 
def success(): 
    if request.method == 'POST': 
        f = request.files['file'] 
        f.save(f.filename)
        prompt = request.form.get('prompt')
        #print(prompt)
        responsetext = process_image(f.filename, model, tokenizer, processor, prompt)
        os.remove(f.filename)
        print(responsetext)
        
        return jsonify(response=responsetext)

if __name__ == '__main__':
	app.run(debug=True, host='0.0.0.0')
