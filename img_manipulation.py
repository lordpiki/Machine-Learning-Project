import numpy as np
from PIL import Image, ImageOps

def apply_scaling(img, t):
    """
    Applies scaling to the input image based on the value of t.
    The output image will be scaled while maintaining the aspect ratio.
    
    Args:
        img (PIL.Image.Image): The input image (28x28 pixels).
        t (float): The scaling factor, ranging from 0 to 1.
        
    Returns:
        PIL.Image.Image: The scaled image.
    """
    width, height = img.size
    assert width == 28 and height == 28, "Input image must be 28x28 pixels."
    
    # Calculate the new dimensions while maintaining the aspect ratio
    new_width = int(width * (1 + t))
    new_height = int(height * (1 + t))
    
    # Resize the image
    scaled_img = img.resize((new_width, new_height), resample=Image.BICUBIC)
    
    # Crop the image back to 28x28 pixels
    left = (new_width - width) // 2
    top = (new_height - height) // 2
    right = left + width
    bottom = top + height
    scaled_img = scaled_img.crop((left, top, right, bottom))
    
    return scaled_img

def apply_noise(img, t):
    """
    Applies noise to the input image based on the value of t.
    
    Args:
        img (PIL.Image.Image): The input image.
        t (float): The noise level, ranging from 0 to 1.
        
    Returns:
        PIL.Image.Image: The image with noise applied.
    """
    t = abs(t)
    pixels = np.array(img)
    noise = np.random.normal(0, t * 255, pixels.shape)
    noisy_pixels = pixels + noise
    noisy_pixels = noisy_pixels.clip(0, 255).astype(np.uint8)
    noisy_img = Image.fromarray(noisy_pixels)
    return noisy_img

def apply_rotation(img, t):
    """
    Applies rotation to the input image based on the value of t.
    
    Args:
        img (PIL.Image.Image): The input image.
        t (float): The rotation angle, ranging from 0 to 1. The actual angle will be t * 360 degrees.
        
    Returns:
        PIL.Image.Image: The rotated image.
    """
    angle = t * 360
    rotated_img = img.rotate(angle, resample=Image.BICUBIC, expand=False)
    return rotated_img

def apply_horizontal_shift(img, t):
    """
    Applies horizontal shifting to the input image based on the value of t.
    The output image will remain 28x28 pixels.
    
    Args:
        img (PIL.Image.Image): The input image (28x28 pixels).
        t (float): The horizontal shift factor, ranging from -1 to 1.
        
    Returns:
        PIL.Image.Image: The shifted image (28x28 pixels).
    """
    width, height = img.size
    assert width == 28 and height == 28, "Input image must be 28x28 pixels."
    
    shift_amount = int(t * width)
    
    # Create a new image with transparent background
    shifted_img = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    
    # Calculate the region to copy from the original image
    if shift_amount > 0:
        src_x = 0
        src_width = width - shift_amount
        dst_x = shift_amount
    else:
        src_x = -shift_amount
        src_width = width + shift_amount
        dst_x = 0
    
    # Copy the relevant region from the original image to the new image
    src_region = img.crop((src_x, 0, src_x + src_width, height))
    shifted_img.paste(src_region, (dst_x, 0))
    
    return shifted_img

def apply_vertical_shift(img, t):
    """
    Applies vertical shifting to the input image based on the value of t.
    The output image will remain 28x28 pixels.
    
    Args:
        img (PIL.Image.Image): The input image (28x28 pixels).
        t (float): The vertical shift factor, ranging from -1 to 1.
        
    Returns:
        PIL.Image.Image: The shifted image (28x28 pixels).
    """
    width, height = img.size
    assert width == 28 and height == 28, "Input image must be 28x28 pixels."
    
    shift_amount = int(t * height)
    
    # Create a new image with transparent background
    shifted_img = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    
    # Calculate the region to copy from the original image
    if shift_amount > 0:
        src_y = 0
        src_height = height - shift_amount
        dst_y = shift_amount
    else:
        src_y = -shift_amount
        src_height = height + shift_amount
        dst_y = 0
    
    # Copy the relevant region from the original image to the new image
    src_region = img.crop((0, src_y, width, src_y + src_height))
    shifted_img.paste(src_region, (0, dst_y))
    
    return shifted_img

def randomize_image(img, range):
    t = np.random.uniform(0.1, 0.15)
    img = apply_noise(img, t)
    t = np.random.uniform(-range * 0.3, range * 0.3)
    img = apply_rotation(img, t)
    t = np.random.uniform(-range * 1.3, range * 1.3)
    img = apply_horizontal_shift(img, t)
    t = np.random.uniform(-range * 1.3, range * 1.3)
    img = apply_vertical_shift(img, t)
    t = np.random.uniform(-range , range)
    img = apply_scaling(img, t)
    return img


if __name__ == "__main__":
    
    # Example usage
    image = Image.open("my_digits/1.png")

    # image = randomize_image(image, 0.2)
    image = apply_scaling(image, 0.2)
    image.show()